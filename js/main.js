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
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if (!href) return;
    const aPath = href.replace(/\/index\.html$/, '/');
    if (aPath === path || (aPath !== '/' && path.startsWith(aPath))) {
      a.classList.add('active');
    }
  });

  /* ── Mobile nav toggle ───────────────────── */
  const toggle = document.querySelector('[data-nav-toggle]');
  const links = document.querySelector('.nav-links');
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
