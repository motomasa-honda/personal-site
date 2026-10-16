/* ============================================================
   RM-Engineering — Site-wide JS
   ============================================================
   - Active nav link 自動付与
   - Mobile nav toggle
   - Blog filter (blog.html でのみ意味を持つ)
   ============================================================ */

(() => {
  'use strict';

  /* ── Active nav link by current path ─────── */
  const path = location.pathname.replace(/\/index\.html$/, '/');

  /* Public pages share a compact personal-site navigation. */
  const primaryNav = document.querySelector('nav[aria-label="Primary"]:not(.site-nav)');
  if (primaryNav) {
    const logo = primaryNav.querySelector('.nav-logo');
    if (logo) {
      logo.textContent = 'motomasa.works';
      logo.setAttribute('aria-label', '本多 基真 ホーム');
      logo.href = '/';
    }
    const list = primaryNav.querySelector('.nav-links');
    if (list) {
      const links = [
        ['/apps.html', 'つくったもの'],
        ['/blog.html', '技術記事'],
        ['/about.html', '自己紹介'],
        ['/contact.html', '連絡する'],
      ];
      list.replaceChildren(...links.map(([href, label]) => {
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = href;
        link.textContent = label;
        if (href === '/contact.html') link.className = 'btn btn-primary';
        item.append(link);
        return item;
      }));
    }
  }

  document.querySelectorAll('.foot-logo').forEach((el) => { el.textContent = '本多 基真 / motomasa.works'; });
  document.querySelectorAll('.foot-copy').forEach((el) => { el.textContent = '© 2026 本多 基真'; });
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if (!href) return;
    const aPath = href.replace(/\/index\.html$/, '/');
    if (aPath === path || (aPath !== '/' && path.startsWith(aPath))) {
      a.classList.add('active');
    }
  });

  /* ── Mobile nav toggle ───────────────────── */
  const activeNav = document.querySelector('nav.site-nav') || primaryNav;
  const toggle = activeNav?.querySelector('[data-nav-toggle]');
  const links = activeNav?.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  /* ── Blog filter ─────────────────────────── */
  const filterButtons = document.querySelectorAll('[data-filter]');
  if (filterButtons.length) {
    filterButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tag = btn.dataset.filter;
        filterButtons.forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('[data-tags]').forEach(card => {
          const tags = (card.dataset.tags || '').toLowerCase();
          card.style.display =
            (tag === 'all' || tags.includes(tag.toLowerCase())) ? '' : 'none';
        });
      });
    });
  }
})();
