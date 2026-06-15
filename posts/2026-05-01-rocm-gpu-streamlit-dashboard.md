---
title: "ROCm環境のAMD GPU（RX 7900 XTX）をStreamlitでリアルタイム監視するダッシュボードを作った"
emoji: "🎮"
type: "tech"
topics: ["streamlit", "rocm", "amd", "python", "ubuntu"]
published: true
publication_date: "2026-05-01"
---

## TL;DR
- AMD RX 7900 XTX + ROCm環境でGPU使用率・VRAM・温度・電力をリアルタイム監視
- `rocm-smi --json` でGPU情報を取得、`psutil` でCPU・メモリを取得
- Streamlitの `st.empty()` + `while True` でタスクマネージャー風のUIを実現
- CPU温度は `lm-sensors` の `k10temp`（Tctl）を直接読まないと誤った値を拾う罠がある
- Mac（192.0.2.1）からUbuntu（192.0.2.2）に `http://192.0.2.2:8502` でアクセス

---

## なぜ作ったか

LLM推論ジョブ（deepseek-r1:70b）をUbuntuのRX 7900 XTXで回しているとき、「今GPUどのくらい使ってるんだろう？」をターミナルで `rocm-smi` 叩くたびに確認するのがしんどくなった。WindowsのタスクマネージャーみたいなUIをMacブラウザから見たくて作った。

構成はシンプルで：
- Ubuntu 24.04（Ryzen 9 9950X / RX 7900 XTX 24GB）がサーバー
- Mac mini M4がクライアント（ブラウザでアクセス）
- Ethernet直結（192.0.2.x）

---

## GPU情報の取得：rocm-smi --json

NVIDIAなら `nvidia-smi` 一発だが、AMD ROCm環境では `rocm-smi` を使う。`--json` フラグでパースしやすい形式で取れる。

```bash
rocm-smi --showuse --showmeminfo vram --showtemp --showpower --json
```

返ってくるJSONはこんな感じ：

```json
{
  "card0": {
    "GPU use (%)": "100",
    "VRAM Total Used Memory (B)": "25677897728",
    "VRAM Total Memory (B)": "25769803776",
    "Temperature (Sensor edge) (C)": "62",
    "Average Graphics Package Power (W)": "323"
  }
}
```

Pythonでのパース：

```python
import subprocess, json

def get_gpu_info():
    r = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram",
         "--showtemp", "--showpower", "--json"],
        capture_output=True, text=True, timeout=5
    )
    data = json.loads(r.stdout)
    card_key = next((k for k in data if k.startswith("card")), None)
    card = data[card_key]

    gpu_use = int(float(str(card.get("GPU use (%)", 0)).replace("%", "")))
    mem_use_gb = int(card.get("VRAM Total Used Memory (B)", 0)) / (1024**3)
    mem_total_gb = int(card.get("VRAM Total Memory (B)", 25769803776)) / (1024**3)
    temp = float(str(card.get("Temperature (Sensor edge) (C)", 0)))
    power = float(str(card.get("Average Graphics Package Power (W)", 0)))

    return {
        "gpu_use": gpu_use,
        "mem_use_gb": round(mem_use_gb, 2),
        "mem_total_gb": round(mem_total_gb, 1),
        "mem_pct": int(mem_use_gb / mem_total_gb * 100),
        "temp": int(temp),
        "power": int(power),
    }
```

キーの名前はROCmのバージョンで微妙に変わることがあるので、複数のキー名を試すようにしておくと安全。

---

## CPU温度の罠：psutilで92°Cが出た

`psutil.sensors_temperatures()` でCPU温度を取ると、Ryzen 9950Xで **92°C** が出た。実際にLLM推論中でもCPU使用率は6%程度なのに異常に高い。

原因は `psutil` がVRMや電源系センサーの値も拾ってしまっていたこと。`sensors` コマンドで確認すると：

```bash
$ sensors | grep -E "Tctl|Tccd|temp"
temp1:        +42.0°C  (マザーボード系)
k10temp-pci-00c3
Tctl:         +62.2°C  ← これが本物のCPU温度
Tccd1:        +62.1°C
Tccd2:        +28.4°C  ← 使われていないダイは低い
```

**Tctl（62°C）が正しい値**で、92°CはVRM等の別センサー。

修正後のコード：

```python
def get_cpu_temp():
    import subprocess, json
    try:
        r = subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=5)
        data = json.loads(r.stdout)
        # k10temp チップの最初のinput値を使う
        for chip, vals in data.items():
            if "k10temp" in chip:
                for sensor, sv in vals.items():
                    if isinstance(sv, dict):
                        for k, v in sv.items():
                            if "input" in k and isinstance(v, (int, float)):
                                return round(v, 1)
    except:
        pass
    return None
```

`lm-sensors` のインストールと初期化も必要：

```bash
sudo apt-get install -y lm-sensors
sudo sensors-detect --auto
sudo modprobe k10temp
```

Ryzen 9000系は `sensors-detect` で「Sorry, no sensors were detected」と出ることがあるが、`k10temp` モジュールを手動でロードすれば取れる。

---

## StreamlitでタスクマネージャーUIを作る

`st.empty()` + `while True` + `time.sleep()` の組み合わせで、ページをリロードせずにリアルタイム更新できる。

```python
import streamlit as st
import time

placeholder = st.empty()
REFRESH_SEC = 2

while True:
    gpu = get_gpu_info()
    cpu_pct = psutil.cpu_percent(interval=0.5)
    # ...データ取得

    with placeholder.container():
        # UIを丸ごと描き直す
        st.markdown(f"GPU使用率: {gpu['gpu_use']}%")
        # ...

    time.sleep(REFRESH_SEC)
```

時系列グラフは `collections.deque` で直近150サンプル（5分）を保持して `st.line_chart()` に渡す：

```python
from collections import deque

if "history" not in st.session_state:
    st.session_state.history = {
        "time":     deque(maxlen=150),  # 2秒×150 = 5分
        "gpu_use":  deque(maxlen=150),
        "gpu_temp": deque(maxlen=150),
        # ...
    }
```

---

## 起動・外部公開

```bash
source ~/ai-env/bin/activate
nohup ~/ai-env/bin/streamlit run ~/kaggle_pipeline/gpu_dashboard.py \
  --server.port 8502 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > ~/kaggle_pipeline/dashboard.log 2>&1 &
```

`--server.address 0.0.0.0` でLAN内の他マシンからアクセス可能になる。
Mac側のブラウザから `http://192.0.2.2:8502` でアクセス。

`nohup` + `source` を1行で書くと環境変数が引き継がれずに `終了 127` になることがある。`~/ai-env/bin/streamlit` のようにフルパスで指定するのが確実。

---

## 実際に動かしてわかったこと（deepseek-r1:70b推論中）

- GPU使用率: **100%**
- VRAM: **23.9 GB / 24 GB（99%）**
- GPU温度: **62°C**（余裕あり）
- GPU電力: **323W**（ROCmで300W制限かけてたはずが少し超えてる）
- CPU使用率: **6.6%**（推論はほぼGPUで完結）
- CPU温度: **62°C**（Tctl、正常範囲）

70Bモデルの推論中はVRAMをほぼ全部使い切る。24GBギリギリ。

---

## 学んだこと
- `rocm-smi --json` のキー名はROCmバージョンで変わることがある。複数のキー名でフォールバックを書いておくべき
- `psutil` の温度取得はAMD環境では信頼できない。`sensors -j` + `k10temp` チップを直接狙う
- Ryzen 9000系は `sensors-detect` で検出されなくてもパニックにならない。`modprobe k10temp` で解決
- Streamlitの `while True` ループは `st.empty()` と組み合わせると意外とスムーズに動く
- `nohup streamlit run` はフルパス指定が安全

## 参考
- ROCm公式ドキュメント: https://rocm.docs.amd.com/
- lm-sensors: https://github.com/lm-sensors/lm-sensors
- 構成: Mac mini M4 (192.0.2.1) ↔ Ubuntu 24.04 RX 7900 XTX (192.0.2.2)
- ファイルパス: `~/kaggle_pipeline/gpu_dashboard.py`、port 8502
