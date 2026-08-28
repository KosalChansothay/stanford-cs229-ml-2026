document.addEventListener('DOMContentLoaded', function () {
  var currentPath = window.location.pathname.split('/').pop() || 'index.html';
  var navLinks = document.querySelectorAll('.sidebar-nav a');
  navLinks.forEach(function (link) {
    var href = link.getAttribute('href');
    if (href === currentPath || (currentPath === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
});