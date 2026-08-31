/* ============================================================
   Academic Pages — Shared Client Script
   Dark Mode System / Match & Toggle, Markdown Parser, MathJax
   ============================================================ */

/* 1. Synchronous Immediate Theme Initialization (Prevents FOUC) */
(function initTheme() {
  try {
    var stored = localStorage.getItem('theme');
    if (stored === 'dark' || stored === 'light') {
      document.documentElement.setAttribute('data-theme', stored);
    } else {
      var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    }
  } catch (e) {
    /* Fallback if localStorage is inaccessible */
  }
})();

/* 2. Real-time OS System Theme Listener */
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
    try {
      if (!localStorage.getItem('theme')) {
        var newTheme = e.matches ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        var btn = document.querySelector('.theme-toggle-btn');
        if (btn) {
          var next = newTheme === 'dark' ? 'light' : 'dark';
          btn.setAttribute('aria-label', 'Switch to ' + next + ' mode');
          btn.setAttribute('title', 'Switch to ' + next + ' mode');
        }
        document.dispatchEvent(new CustomEvent('theme:changed', { detail: { theme: newTheme } }));
      }
    } catch (err) {}
  });
}

document.addEventListener('DOMContentLoaded', function () {
  /* Active navigation highlighting */
  var currentPath = window.location.pathname.replace(/\\/g, '/');
  var navLinks = document.querySelectorAll('.sidebar-nav a');
  navLinks.forEach(function (link) {
    var linkPath = new URL(link.href, window.location.href).pathname.replace(/\\/g, '/');
    var isCourseBranch = linkPath.endsWith('/courses/index.html') && currentPath.indexOf('/courses/') !== -1;
    if (linkPath === currentPath || (currentPath.endsWith('/') && linkPath === currentPath + 'index.html') || isCourseBranch) {
      link.classList.add('active');
    }
  });

  var sidebar = document.querySelector('.sidebar');
  if (sidebar) {
    /* Theme Toggle Switch (Sun / Moon) */
    var themeBtn = document.createElement('button');
    themeBtn.className = 'theme-toggle-btn';
    themeBtn.type = 'button';
    themeBtn.innerHTML = '<svg class="icon-moon" aria-hidden="true" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>' +
                         '<svg class="icon-sun" aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';

    function updateThemeBtnAria() {
      var current = document.documentElement.getAttribute('data-theme') ||
                    (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      themeBtn.setAttribute('aria-label', 'Switch to ' + next + ' mode');
      themeBtn.setAttribute('title', 'Switch to ' + next + ' mode');
    }

    updateThemeBtnAria();

    themeBtn.addEventListener('click', function () {
      var current = document.documentElement.getAttribute('data-theme') ||
                    (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try {
        localStorage.setItem('theme', next);
      } catch (e) {}
      updateThemeBtnAria();
      document.dispatchEvent(new CustomEvent('theme:changed', { detail: { theme: next } }));
    });

    var social = sidebar.querySelector('.sidebar-social');
    if (social) {
      sidebar.insertBefore(themeBtn, social);
    } else {
      sidebar.appendChild(themeBtn);
    }

    /* Mobile Menu Toggle */
    var toggle = document.createElement('button');
    toggle.className = 'sidebar-toggle';
    toggle.type = 'button';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-controls', 'site-navigation');
    toggle.textContent = 'Menu';
    sidebar.appendChild(toggle);
    var nav = document.querySelector('.sidebar-nav');
    if (nav) nav.id = 'site-navigation';
    toggle.addEventListener('click', function () {
      var isOpen = sidebar.classList.toggle('nav-open');
      toggle.setAttribute('aria-expanded', String(isOpen));
    });
  }

  /* Note Search & Filtering */
  var filter = document.querySelector('[data-note-filter]');
  if (filter) {
    var notes = document.querySelectorAll('[data-note]');
    var emptyState = document.querySelector('[data-filter-empty]');
    filter.addEventListener('input', function () {
      var query = filter.value.toLowerCase().trim();
      var visible = 0;
      notes.forEach(function (note) {
        var matches = note.textContent.toLowerCase().indexOf(query) !== -1;
        note.hidden = !matches;
        if (matches) visible += 1;
      });
      if (emptyState) emptyState.hidden = visible !== 0;
    });
  }

  /* Dynamic Markdown Loading for Lecture Notes */
  var markdownPage = document.querySelector('[data-markdown]');
  var markdownContent = document.querySelector('[data-markdown-content]');
  if (markdownPage && markdownContent && window.marked) {
    fetch(markdownPage.dataset.markdown)
      .then(function (response) {
        if (!response.ok) throw new Error('Unable to load lecture notes');
        return response.text();
      })
      .then(function (markdown) {
        markdownContent.innerHTML = marked.parse(markdown);
        var h1 = markdownContent.querySelector('h1');
        if (h1) h1.remove();
        var h2 = markdownContent.querySelector('h2');
        if (h2 && h2.textContent && h2.textContent.indexOf('Lecture 2:') !== -1) {
          h2.remove();
        }
        markdownPage.querySelectorAll('.card').forEach(function (card) {
          card.remove();
        });

        /* Syntax highlighting + copy buttons for fenced code blocks. */
        if (window.hljs) {
          markdownContent.querySelectorAll('pre code').forEach(function (block) {
            if (window.hljs) hljs.highlightElement(block);
          });
        }
        markdownContent.querySelectorAll('pre').forEach(function (pre) {
          if (pre.querySelector('.code-copy-btn')) return;
          var btn = document.createElement('button');
          btn.className = 'code-copy-btn';
          btn.type = 'button';
          btn.setAttribute('aria-label', 'Copy code to clipboard');
          btn.textContent = 'Copy';
          btn.addEventListener('click', function () {
            var code = pre.querySelector('code');
            var text = code ? code.innerText : pre.textContent;
            var done = function () {
              btn.textContent = 'Copied!';
              setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
            } else {
              fallbackCopy(text);
              done();
            }
          });
          function fallbackCopy(text) {
            var ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand('copy'); } catch (e) { /* noop */ }
            document.body.removeChild(ta);
          }
          pre.appendChild(btn);
        });

        if (window.MathJax) {
          var typeset = function () {
            return MathJax.typesetPromise ? MathJax.typesetPromise([markdownContent]) : null;
          };
          if (MathJax.startup && MathJax.startup.promise) {
            MathJax.startup.promise.then(typeset);
          } else {
            typeset();
          }
        }

        /* Notify page-specific chart scripts that the markdown is in the DOM. */
        document.dispatchEvent(new CustomEvent('markdown:rendered'));

        /* Synchronize chart theme immediately on initial load */
        setTimeout(function () {
          var currentTheme = document.documentElement.getAttribute('data-theme') ||
            (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
          updateAllChartsTheme(currentTheme);
        }, 120);
      })
      .catch(function (error) {
        console.warn(error.message);
      });
  }

  function updateAllChartsTheme(theme) {
    var isDark = theme === 'dark';
    var charts = document.querySelectorAll('.plotly-chart');
    charts.forEach(function (chartEl) {
      if (window.Plotly && chartEl.data) {
        try {
          var update = {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font.color': isDark ? '#cbd5dc' : '#26343b',
            'xaxis.gridcolor': isDark ? '#282e33' : '#e5e5e5',
            'yaxis.gridcolor': isDark ? '#282e33' : '#e5e5e5',
            'xaxis.tickfont.color': isDark ? '#8e9aa2' : '#637179',
            'yaxis.tickfont.color': isDark ? '#8e9aa2' : '#637179',
            'xaxis.title.font.color': isDark ? '#f0f3f6' : '#26343b',
            'yaxis.title.font.color': isDark ? '#f0f3f6' : '#26343b',
            'title.font.color': isDark ? '#f0f3f6' : '#173f5f'
          };
          if (chartEl.layout && chartEl.layout.yaxis2) {
            update['yaxis2.gridcolor'] = isDark ? '#282e33' : '#e5e5e5';
          }
          Plotly.relayout(chartEl, update);
        } catch (err) {
          /* noop */
        }
      }
    });
  }

  /* Listen for Theme changes to update Plotly charts dynamically */
  document.addEventListener('theme:changed', function (e) {
    updateAllChartsTheme(e.detail.theme);
  });
});