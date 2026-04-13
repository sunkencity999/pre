// PRE Web GUI — Theme switching with dropdown selector

const Themes = (() => {
  const STORAGE_KEY = 'pre-theme';
  const THEMES = [
    { id: 'dark',       name: 'Dark',       icon: '◗' },
    { id: 'light',      name: 'Light',      icon: '◑' },
  ];

  function get() {
    return localStorage.getItem(STORAGE_KEY) || 'dark';
  }

  function set(theme) {
    if (!THEMES.find(t => t.id === theme)) return;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
    updateDropdown();
  }

  function updateDropdown() {
    const current = get();
    const meta = THEMES.find(t => t.id === current) || THEMES[0];
    const label = document.getElementById('theme-current-label');
    if (label) label.textContent = meta.name;
    document.querySelectorAll('.theme-option').forEach(opt => {
      opt.classList.toggle('active', opt.dataset.theme === current);
    });
  }

  function init() {
    // Build dropdown
    const container = document.querySelector('.theme-switcher');
    if (!container) return;

    container.innerHTML = `
      <div class="theme-dropdown-wrapper">
        <button id="theme-toggle-btn" class="theme-toggle-btn" title="Switch theme">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2m-8-10H2m20 0h-2m-2.93-7.07l-1.41 1.41m-9.32 9.32l-1.41 1.41m0-12.14l1.41 1.41m9.32 9.32l1.41 1.41"/></svg>
          <span id="theme-current-label">Dark</span>
          <svg class="theme-chevron" width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M4.427 7.427l3.396 3.396a.25.25 0 00.354 0l3.396-3.396A.25.25 0 0011.396 7H4.604a.25.25 0 00-.177.427z"/></svg>
        </button>
        <div id="theme-dropdown" class="theme-dropdown hidden">
          ${THEMES.map(t => `
            <button class="theme-option" data-theme="${t.id}">
              <span class="theme-option-icon">${t.icon}</span>
              <span>${t.name}</span>
            </button>
          `).join('')}
        </div>
      </div>
    `;

    // Apply saved theme
    set(get());

    // Bind toggle
    const btn = document.getElementById('theme-toggle-btn');
    const dropdown = document.getElementById('theme-dropdown');

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      dropdown.classList.toggle('hidden');
    });

    // Bind options
    dropdown.addEventListener('click', (e) => {
      const opt = e.target.closest('.theme-option');
      if (!opt) return;
      set(opt.dataset.theme);
      dropdown.classList.add('hidden');
    });

    // Close on outside click
    document.addEventListener('click', () => dropdown.classList.add('hidden'));
  }

  return { init, get, set };
})();
