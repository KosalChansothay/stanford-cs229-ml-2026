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
      })
      .catch(function (error) {
        console.warn(error.message);
      });
  }
});