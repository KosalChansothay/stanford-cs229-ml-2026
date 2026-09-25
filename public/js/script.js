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
    var isCourseBranch = linkPath.endsWith('/courses/index.html') && currentPath.indexOf('/courses/') !== -1 && !currentPath.endsWith('/courses/index.html');
    if (linkPath === currentPath || (currentPath.endsWith('/') && linkPath === currentPath + 'index.html')) {
      link.classList.add('active');
    }
  });

  var sidebar = document.querySelector('.sidebar');
  if (sidebar) {
    var rawPath = window.location.pathname.replace(/\\/g, '/');
    var isKhmer = document.documentElement.lang === 'km' || rawPath.indexOf('/km/') !== -1 || rawPath.endsWith('/km') || rawPath.endsWith('/km/');

    /* Find root-relative prefix from existing home link in sidebar */
    var homeLink = document.querySelector('.sidebar-nav a[href*="index.html"]');
    var rootPrefix = '';
    if (homeLink) {
      var rawHref = homeLink.getAttribute('href');
      rootPrefix = rawHref.replace(/index\.html$/, '');
    }

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

    /* Language Switcher Button (English <-> ភាសាខ្មែរ) */
    function resolveLangSwitchUrl() {
      if (isKhmer) {
        var target = rawPath.replace(/\/km(\/|$)/, '/');
        if (target === '' || target.endsWith('/')) {
          target += 'index.html';
        }
        return target + window.location.search + window.location.hash;
      } else {
        var segments = rawPath.split('/').filter(Boolean);
        var repoPrefix = '';
        var knownRoots = ['courses', 'notes', 'img', 'css', 'js', 'km'];
        if (segments.length > 0 && knownRoots.indexOf(segments[0]) === -1 && !segments[0].endsWith('.html')) {
          repoPrefix = '/' + segments[0];
          segments.shift();
        }
        var innerPath = segments.join('/');
        if (!innerPath || innerPath === 'index.html') {
          innerPath = 'index.html';
        }
        return repoPrefix + '/km/' + innerPath + window.location.search + window.location.hash;
      }
    }

    var langBtn = document.createElement('a');
    langBtn.className = 'lang-toggle-btn';
    langBtn.href = resolveLangSwitchUrl();
    langBtn.setAttribute('aria-label', isKhmer ? 'Switch to English' : 'ប្តូរទៅជាភាសាខ្មែរ (Switch to Khmer)');
    langBtn.setAttribute('title', isKhmer ? 'Switch to English' : 'ប្តូរទៅជាភាសាខ្មែរ (Switch to Khmer)');
    langBtn.innerHTML = '<svg class="lang-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>' +
                        '<span>' + (isKhmer ? 'English' : 'ខ្មែរ') + '</span>';

    langBtn.addEventListener('click', function () {
      try {
        localStorage.setItem('preferred-lang', isKhmer ? 'en' : 'km');
      } catch (err) {}
    });

    /* Course Accordions Generator */
    var coursesData = [
      {
        code: 'CS229',
        badgeClass: 'badge-cs229',
        nameEn: 'CS229: Machine Learning',
        nameKm: 'CS229: ការរៀនរបស់ម៉ាស៊ីន',
        folder: 'cs229',
        countText: '14',
        lectures: [
          { titleEn: 'Lec 1: Introduction', titleKm: 'មេរៀនទី ១: សេចក្តីផ្តើម', file: 'lecture-1.html' },
          { titleEn: 'Lec 2: Supervised Setup', titleKm: 'មេរៀនទី ២: Supervised Setup', file: 'lecture-2.html' },
          { titleEn: 'Lec 3: Logistic Regression', titleKm: 'មេរៀនទី ៣: Logistic Regression', file: 'lecture-3.html' },
          { titleEn: 'Lec 4: GLMs & Exponential', titleKm: 'មេរៀនទី ៤: GLMs & Exponential', file: 'lecture-4.html' },
          { titleEn: 'Lec 5: GDA & Naive Bayes', titleKm: 'មេរៀនទី ៥: GDA & Naive Bayes', file: 'lecture-5.html' },
          { titleEn: 'Lec 6: Bias-Variance & Advice', titleKm: 'មេរៀនទី ៦: Bias-Variance', file: 'lecture-6.html' },
          { titleEn: 'Lec 7: Architecture', titleKm: 'មេរៀនទី ៧: Architecture', file: 'lecture-7.html' },
          { titleEn: 'Lec 8: Backpropagation', titleKm: 'មេរៀនទី ៨: Backpropagation', file: 'lecture-8.html' },
          { titleEn: 'Lec 9: K-Means & GMM', titleKm: 'មេរៀនទី ៩: K-Means & GMM', file: 'lecture-9.html' },
          { titleEn: 'Lec 10: EM & PCA', titleKm: 'មេរៀនទី ១០: EM & PCA', file: 'lecture-10.html' },
          { titleEn: 'Lec 11: Diffusion Models', titleKm: 'មេរៀនទី ១១: Diffusion Models', file: 'lecture-11.html' },
          { titleEn: 'Lec 12: Foundation & LoRA', titleKm: 'មេរៀនទី ១២: Foundation & LoRA', file: 'lecture-12.html' },
          { titleEn: 'Lec 13: Contrastive & RAG', titleKm: 'មេរៀនទី ១៣: Contrastive & RAG', file: 'lecture-13.html' },
          { titleEn: 'Lec 14: Transformers', titleKm: 'មេរៀនទី ១៤: Transformers', file: 'lecture-14.html' }
        ]
      },
      {
        code: 'CS336',
        badgeClass: 'badge-cs336',
        nameEn: 'CS336: Language Modeling',
        nameKm: 'CS336: គំរូភាសា (LLM)',
        folder: 'cs336',
        countText: '18',
        lectures: [
          { titleEn: 'Lec 1: Tokenization', titleKm: 'មេរៀនទី ១: Tokenization', file: 'lecture-1.html' },
          { titleEn: 'Lec 2: PyTorch Accounting', titleKm: 'មេរៀនទី ២: PyTorch Accounting', file: 'lecture-2.html' },
          { titleEn: 'Lec 3: Architectures', titleKm: 'មេរៀនទី ៣: Architectures', file: 'lecture-3.html' },
          { titleEn: 'Lec 4: Attention & MoE', titleKm: 'មេរៀនទី ៤: Attention & MoE', file: 'lecture-4.html' },
          { titleEn: 'Lec 5: GPUs & FlashAttention', titleKm: 'មេរៀនទី ៥: FlashAttention', file: 'lecture-5.html' },
          { titleEn: 'Lec 6: Kernels & Triton', titleKm: 'មេរៀនទី ៦: Kernels & Triton', file: 'lecture-6.html' },
          { titleEn: 'Lec 7: Parallelism Foundations', titleKm: 'មេរៀនទី ៧: Parallelism', file: 'lecture-7.html' },
          { titleEn: 'Lec 8: ZeRO & FSDP', titleKm: 'មេរៀនទី ៨: ZeRO & FSDP', file: 'lecture-8.html' },
          { titleEn: 'Lec 9: Scaling Laws', titleKm: 'មេរៀនទី ៩: Scaling Laws', file: 'lecture-9.html' },
          { titleEn: 'Lec 10: Inference & KV Cache', titleKm: 'មេរៀនទី ១០: KV Cache', file: 'lecture-10.html' },
          { titleEn: 'Lec 11: Scaling & Optimizers', titleKm: 'មេរៀនទី ១១: Optimizers', file: 'lecture-11.html' },
          { titleEn: 'Lec 12: Evaluation & Leakage', titleKm: 'មេរៀនទី ១២: Evaluation', file: 'lecture-12.html' },
          { titleEn: 'Lec 13: Pretraining Data', titleKm: 'មេរៀនទី ១៣: Pretraining Data', file: 'lecture-13.html' },
          { titleEn: 'Lec 14: Data Preprocessing', titleKm: 'មេរៀនទី ១៤: Data Preprocessing', file: 'lecture-14.html' },
          { titleEn: 'Lec 15: Post-Training (SFT/DPO)', titleKm: 'មេរៀនទី ១៥: SFT & DPO', file: 'lecture-15.html' },
          { titleEn: 'Lec 16: Post-Training (RLVR)', titleKm: 'មេរៀនទី ១៦: RLVR & GRPO', file: 'lecture-16.html' },
          { titleEn: 'Lec 17: Alignment & Multimodal', titleKm: 'មេរៀនទី ១៧: Alignment', file: 'lecture-17.html' },
          { titleEn: 'Guest: Systems & Arch', titleKm: 'Guest: Systems & Arch', file: 'lecture-guest.html' }
        ]
      },
      {
        code: 'CS231N',
        badgeClass: 'badge-cs231n',
        nameEn: 'CS231N: Computer Vision',
        nameKm: 'CS231N: គំហើញកុំព្យូទ័រ',
        folder: 'cs231n',
        countText: '18',
        lectures: [
          { titleEn: 'Lec 1: Intro & History', titleKm: 'មេរៀនទី ១: Intro & History', file: 'lecture-1.html' },
          { titleEn: 'Lec 2: Classifiers & Loss', titleKm: 'មេរៀនទី ២: Classifiers & Loss', file: 'lecture-2.html' },
          { titleEn: 'Lec 3: Optimization', titleKm: 'មេរៀនទី ៣: Optimization', file: 'lecture-3.html' },
          { titleEn: 'Lec 4: Backprop & Nets', titleKm: 'មេរៀនទី ៤: Backprop & Nets', file: 'lecture-4.html' },
          { titleEn: 'Lec 5: CNNs & Convolutions', titleKm: 'មេរៀនទី ៥: CNNs', file: 'lecture-5.html' },
          { titleEn: 'Lec 6: Architectures & ResNet', titleKm: 'មេរៀនទី ៦: ResNet', file: 'lecture-6.html' },
          { titleEn: 'Lec 7: RNNs & LSTMs', titleKm: 'មេរៀនទី ៧: RNNs & LSTMs', file: 'lecture-7.html' },
          { titleEn: 'Lec 8: Vision Transformers', titleKm: 'មេរៀនទី ៨: ViT', file: 'lecture-8.html' },
          { titleEn: 'Lec 9: Detection & Seg', titleKm: 'មេរៀនទី ៩: Detection', file: 'lecture-9.html' },
          { titleEn: 'Lec 10: Video Understanding', titleKm: 'មេរៀនទី ១០: Video', file: 'lecture-10.html' },
          { titleEn: 'Lec 11: Distributed Training', titleKm: 'មេរៀនទី ១១: Distributed', file: 'lecture-11.html' },
          { titleEn: 'Lec 12: Self-Supervised', titleKm: 'មេរៀនទី ១២: Self-Supervised', file: 'lecture-12.html' },
          { titleEn: 'Lec 13: Generative (VAE/GAN)', titleKm: 'មេរៀនទី ១៣: VAE & GAN', file: 'lecture-13.html' },
          { titleEn: 'Lec 14: Diffusion Models', titleKm: 'មេរៀនទី ១៤: Diffusion', file: 'lecture-14.html' },
          { titleEn: 'Lec 15: 3D Vision & NeRF', titleKm: 'មេរៀនទី ១៥: 3D & NeRF', file: 'lecture-15.html' },
          { titleEn: 'Lec 16: Vision-Language (CLIP)', titleKm: 'មេរៀនទី ១៦: CLIP', file: 'lecture-16.html' },
          { titleEn: 'Lec 17: Robot Learning', titleKm: 'មេរៀនទី ១៧: Robot Learning', file: 'lecture-17.html' },
          { titleEn: 'Lec 18: Human-Centered AI', titleKm: 'មេរៀនទី ១៨: Human-Centered AI', file: 'lecture-18.html' }
        ]
      }
    ];

    var coursesContainer = document.createElement('div');
    coursesContainer.className = 'sidebar-courses';

    var coursesHeading = document.createElement('div');
    coursesHeading.className = 'nav-label';
    coursesHeading.textContent = isKhmer ? 'វគ្គសិក្សា (Courses)' : 'Courses';
    coursesContainer.appendChild(coursesHeading);

    coursesData.forEach(function (course) {
      var isCurrentCourse = currentPath.indexOf('/' + course.folder + '/') !== -1 || currentPath.indexOf('/' + course.folder) !== -1;

      var accordion = document.createElement('div');
      accordion.className = 'course-accordion';
      if (isCurrentCourse) {
        accordion.classList.add('is-open', 'is-active-course');
      }

      var headerBtn = document.createElement('button');
      headerBtn.type = 'button';
      headerBtn.className = 'course-accordion-header';
      headerBtn.setAttribute('aria-expanded', isCurrentCourse ? 'true' : 'false');

      var titleDiv = document.createElement('div');
      titleDiv.className = 'course-accordion-title';

      var badgeSpan = document.createElement('span');
      badgeSpan.className = 'course-accordion-badge ' + course.badgeClass;
      badgeSpan.textContent = course.code;

      var nameSpan = document.createElement('span');
      nameSpan.textContent = isKhmer ? course.nameKm : course.nameEn;

      titleDiv.appendChild(badgeSpan);
      titleDiv.appendChild(nameSpan);

      var chevron = document.createElement('span');
      chevron.className = 'course-accordion-chevron';
      chevron.innerHTML = '&#9656;';

      headerBtn.appendChild(titleDiv);
      headerBtn.appendChild(chevron);

      headerBtn.addEventListener('click', function () {
        accordion.classList.toggle('is-open');
        headerBtn.setAttribute('aria-expanded', String(accordion.classList.contains('is-open')));
      });

      var list = document.createElement('ul');
      list.className = 'course-lecture-list';

      // Course Overview link
      var overviewLi = document.createElement('li');
      var overviewA = document.createElement('a');
      overviewA.className = 'course-overview-link';
      if (isKhmer) {
        if (course.folder === 'cs229') {
          overviewA.href = rootPrefix + 'courses/cs229/index.html';
        } else {
          overviewA.href = rootPrefix + '../courses/' + course.folder + '/index.html';
        }
      } else {
        overviewA.href = rootPrefix + 'courses/' + course.folder + '/index.html';
      }
      overviewA.textContent = isKhmer ? 'ទិដ្ឋភាពទូទៅនៃ ' + course.code : course.code + ' Overview';
      if (currentPath.indexOf('/' + course.folder + '/index.html') !== -1) {
        overviewA.classList.add('active');
      }
      overviewLi.appendChild(overviewA);
      list.appendChild(overviewLi);

      // Lectures list
      course.lectures.forEach(function (lec) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        if (isKhmer) {
          if (course.folder === 'cs229') {
            a.href = rootPrefix + 'courses/cs229/' + lec.file;
          } else {
            a.href = rootPrefix + '../courses/' + course.folder + '/' + lec.file;
          }
        } else {
          a.href = rootPrefix + 'courses/' + course.folder + '/' + lec.file;
        }
        a.textContent = isKhmer ? lec.titleKm : lec.titleEn;
        if (isCurrentCourse && currentPath.indexOf(lec.file) !== -1) {
          a.classList.add('active');
        }
        li.appendChild(a);
        list.appendChild(li);
      });

      accordion.appendChild(headerBtn);
      accordion.appendChild(list);
      coursesContainer.appendChild(accordion);
    });

    var navEl = sidebar.querySelector('.sidebar-nav');
    if (navEl && navEl.nextSibling) {
      sidebar.insertBefore(coursesContainer, navEl.nextSibling);
    } else {
      sidebar.appendChild(coursesContainer);
    }

    /* Sidebar Bottom & Footer Organization */
    var sidebarBottom = document.createElement('div');
    sidebarBottom.className = 'sidebar-bottom';

    var sidebarTools = document.createElement('div');
    sidebarTools.className = 'sidebar-tools';
    sidebarTools.appendChild(langBtn);
    sidebarTools.appendChild(themeBtn);

    var social = sidebar.querySelector('.sidebar-social');
    if (social) {
      sidebarTools.appendChild(social);
    }

    sidebarBottom.appendChild(sidebarTools);

    var existingFooter = sidebar.querySelector('.sidebar-footer');
    if (existingFooter) {
      sidebarBottom.appendChild(existingFooter);
    }

    sidebar.appendChild(sidebarBottom);

    /* Mobile Sticky Topbar & Off-canvas Drawer */
    var mobileTopbar = document.createElement('header');
    mobileTopbar.className = 'mobile-topbar';

    var topbarLeft = document.createElement('div');
    topbarLeft.className = 'mobile-topbar-left';

    var mobileMenuBtn = document.createElement('button');
    mobileMenuBtn.type = 'button';
    mobileMenuBtn.className = 'mobile-menu-btn';
    mobileMenuBtn.setAttribute('aria-label', isKhmer ? 'បើកម៉ឺនុយ' : 'Open menu');
    mobileMenuBtn.setAttribute('aria-expanded', 'false');
    mobileMenuBtn.innerHTML = '&#9776;';

    var mobileBrand = document.createElement('a');
    mobileBrand.className = 'mobile-topbar-brand';
    mobileBrand.href = homeLink ? homeLink.href : (isKhmer ? 'index.html' : 'index.html');
    mobileBrand.innerHTML = '<img class="mobile-topbar-avatar" src="' + rootPrefix + 'img/profile.jpg" alt="Logo">' +
                            '<span>' + (isKhmer ? 'កំណត់ចំណាំ ML' : 'Stanford ML') + '</span>';

    topbarLeft.appendChild(mobileMenuBtn);
    topbarLeft.appendChild(mobileBrand);

    var topbarRight = document.createElement('div');
    topbarRight.className = 'mobile-topbar-right';

    var mobileLangBtn = langBtn.cloneNode(true);
    var mobileThemeBtn = themeBtn.cloneNode(true);
    mobileThemeBtn.addEventListener('click', function () {
      themeBtn.click();
    });

    topbarRight.appendChild(mobileLangBtn);
    topbarRight.appendChild(mobileThemeBtn);

    mobileTopbar.appendChild(topbarLeft);
    mobileTopbar.appendChild(topbarRight);

    document.body.insertBefore(mobileTopbar, sidebar);

    var backdrop = document.createElement('div');
    backdrop.className = 'sidebar-backdrop';
    document.body.appendChild(backdrop);

    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'sidebar-close-btn';
    closeBtn.setAttribute('aria-label', isKhmer ? 'បិទម៉ឺនុយ' : 'Close menu');
    closeBtn.innerHTML = '&times;';
    sidebar.insertBefore(closeBtn, sidebar.firstChild);

    function openSidebar() {
      document.body.classList.add('sidebar-open');
      mobileMenuBtn.setAttribute('aria-expanded', 'true');
    }

    function closeSidebar() {
      document.body.classList.remove('sidebar-open');
      mobileMenuBtn.setAttribute('aria-expanded', 'false');
    }

    mobileMenuBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (document.body.classList.contains('sidebar-open')) {
        closeSidebar();
      } else {
        openSidebar();
      }
    });

    closeBtn.addEventListener('click', closeSidebar);
    backdrop.addEventListener('click', closeSidebar);

    sidebar.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        if (window.innerWidth < 900) {
          closeSidebar();
        }
      });
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

  /* Listen for Theme changes to update Plotly charts and mobile toggle dynamically */
  document.addEventListener('theme:changed', function (e) {
    updateAllChartsTheme(e.detail.theme);
    var mobileBtn = document.querySelector('.mobile-topbar .theme-toggle-btn');
    if (mobileBtn) {
      var next = e.detail.theme === 'dark' ? 'light' : 'dark';
      mobileBtn.setAttribute('aria-label', 'Switch to ' + next + ' mode');
      mobileBtn.setAttribute('title', 'Switch to ' + next + ' mode');
    }
  });
});