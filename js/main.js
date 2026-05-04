// RM-Engineering — main.js

// Typing effect
const phrases = ['whoami', 'ls -la projects/', 'ollama run qwen3.6:27b', 'python3 supervisor.py'];
let phraseIndex = 0, charIndex = 0, isDeleting = false;

function type() {
  const el = document.getElementById('typed');
  if (!el) return;
  const phrase = phrases[phraseIndex];
  if (isDeleting) {
    el.textContent = phrase.substring(0, charIndex--);
    if (charIndex < 0) { isDeleting = false; phraseIndex = (phraseIndex + 1) % phrases.length; setTimeout(type, 500); return; }
  } else {
    el.textContent = phrase.substring(0, charIndex++);
    if (charIndex > phrase.length) { isDeleting = true; setTimeout(type, 1500); return; }
  }
  setTimeout(type, isDeleting ? 40 : 80);
}
document.addEventListener('DOMContentLoaded', () => setTimeout(type, 800));

// Counter animation
function animateCounters() {
  document.querySelectorAll('.stat-num').forEach(el => {
    const target = +el.dataset.target;
    let current = 0;
    const step = target / 40;
    const timer = setInterval(() => {
      current += step;
      if (current >= target) { el.textContent = target; clearInterval(timer); return; }
      el.textContent = Math.floor(current);
    }, 30);
  });
}

const statsSection = document.querySelector('.stats');
if (statsSection) {
  const observer = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { animateCounters(); observer.disconnect(); }
  }, { threshold: 0.5 });
  observer.observe(statsSection);
}

// Nav toggle
function toggleNav() {
  document.querySelector('.nav-links').classList.toggle('open');
}

// Active nav link
document.querySelectorAll('.nav-links a').forEach(a => {
  if (a.href === window.location.href) a.classList.add('active');
  else a.classList.remove('active');
});
