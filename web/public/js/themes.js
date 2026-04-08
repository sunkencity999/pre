// PRE Web GUI — Theme switching with smooth transitions

const Themes = (() => {
  const STORAGE_KEY = 'pre-theme';
  const THEMES = ['dark', 'light', 'evangelion'];

  function get() {
    return localStorage.getItem(STORAGE_KEY) || 'dark';
  }

  function set(theme) {
    if (!THEMES.includes(theme)) return;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
    updateButtons();
  }

  function updateButtons() {
    const current = get();
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.theme === current);
    });
  }

  function init() {
    // Apply saved theme
    set(get());

    // Bind click handlers
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.addEventListener('click', () => set(btn.dataset.theme));
    });
  }

  return { init, get, set };
})();
