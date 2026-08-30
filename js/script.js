document.addEventListener('DOMContentLoaded', function () {
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
  var toggle = document.createElement('button');
  toggle.className = 'sidebar-toggle';
  toggle.type = 'button';
  toggle.setAttribute('aria-expanded', 'false');
  toggle.setAttribute('aria-controls', 'site-navigation');
  toggle.textContent = 'Menu';
  sidebar.appendChild(toggle);
  document.querySelector('.sidebar-nav').id = 'site-navigation';
  toggle.addEventListener('click', function () {
    var isOpen = sidebar.classList.toggle('nav-open');
    toggle.setAttribute('aria-expanded', String(isOpen));
  });

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
        markdownContent.querySelector('h1')?.remove();
        if (markdownContent.querySelector('h2')?.textContent.indexOf('Lecture 2:') !== -1) {
          markdownContent.querySelector('h2').remove();
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
      })
      .catch(function (error) {
        console.warn(error.message);
      });
  }
});