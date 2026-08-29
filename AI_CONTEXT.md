# AI Context: Machine Learning Note

## Purpose

This repository is a small static academic notes website for Stanford CS229 machine learning lectures. It is inspired by the academicpages GitHub template. The site is intentionally simple: HTML, CSS, JavaScript, Markdown, MathJax, and CDN-loaded Marked.js. There is no npm project, bundler, framework, or build step.

Read this file before changing the project. Prefer the existing patterns and make focused edits.

## Current Source Of Truth

Lecture content is maintained in Markdown files in the `notes/` folder:

- `notes/cs229-lecture-1-study-notes.md` -> `courses/cs229/lecture-1.html`
- `notes/cs229-lecture-2-study-notes.md` -> `courses/cs229/lecture-2.html`
- `notes/cs229-lecture-3-study-notes.md` -> `courses/cs229/lecture-3.html`
- `notes/cs229-lecture-4-study-notes.md` -> `courses/cs229/lecture-4.html`
- `notes/cs229-lecture-5-study-notes.md` -> `courses/cs229/lecture-5.html`
- `notes/cs229-lecture-6-study-notes.md` -> `courses/cs229/lecture-6.html`

Lecture Markdown files must NOT contain bracketed numeric citation references such as `[358, 429]` or `[364]`.
They are leftover transcript timestamps and are stripped before publishing. When adding new lecture notes,
remove them with a regex like `\s*\[\d+(,\s*\d+)*\]` (this safely preserves math like one-hot vectors `[1, 0, 0, 0]`
because those contain zeros and live inside `$...$`).

Edit the Markdown files for lecture content. Do not copy new lecture content into the HTML pages unless a fallback is intentionally being updated.

Markdown supports:

- ATX headings: `## Section`, `### Subsection`
- Unordered and ordered lists
- Fenced code blocks
- Inline math: `$x$`
- Display math: `$$ ... $$`
- Horizontal rules: `---`
- Plotly chart containers: `<div id="plotly-..." class="plotly-chart"></div>` rendered by page-specific scripts in `js/` (see `js/lecture3-charts.js` for the pattern)

Keep the Markdown readable and structured like academic notes: summary, definitions, derivations, algorithms or workflows, examples, reflection questions, and references.

## Page Architecture

Root pages:

- `index.html`: site home and featured lectures
- `about.html`: project description and technology information
- `notes.html`: lecture index with client-side search

Course pages:

- `courses/index.html`: all courses
- `courses/cs229/index.html`: CS229 course home
- `courses/cs229/lecture-1.html`: Lecture 1 shell and Markdown renderer
- `courses/cs229/lecture-2.html`: Lecture 2 shell and Markdown renderer
- `courses/cs229/lecture-3.html`: Lecture 3 shell, Markdown renderer, and Plotly.js charts
- `courses/cs229/lecture-4.html`: Lecture 4 shell and Markdown renderer
- `courses/cs229/lecture-5.html`: Lecture 5 shell and Markdown renderer
- `courses/cs229/lecture-6.html`: Lecture 6 shell and Markdown renderer

Shared assets:

- `css/style.css`: all layout, typography, responsive, academic document, and component styles (including `.plotly-chart` containers)
- `js/script.js`: active navigation, mobile menu, lecture filtering, Markdown loading, MathJax re-typesetting, and the `markdown:rendered` event
- `js/lecture3-charts.js`: page-specific Plotly charts for Lecture 3, drawn after `markdown:rendered` fires
- `img/profile.jpg`: site logo/avatar used in the top navigation bar on every page

## Navigation Layout (Top Bar, Not Sidebar)

The navigation is a **sticky top navbar** (`.sidebar` class kept for historical reasons), not a left sidebar:

- `position: sticky; top: 0` horizontal flex bar: logo image + site name on the left, nav links right-aligned, GitHub icon at the end.
- The avatar is `<img class="sidebar-avatar" src=".../img/profile.jpg">` (44px circle, `object-fit: cover`), NOT an emoji div.
- `.sidebar-bio` and `.sidebar-footer` are hidden via CSS.
- `body` must NOT have `display: flex` — that stretches the navbar to full page height.
- `.main` is centered with `max-width: var(--max-width)` and no left margin.
- On `<= 768px` the nav collapses into a "Menu" dropdown toggle (created by `js/script.js`).

## SEO / Meta Tags Contract

Every HTML page must include, after the description tag:

```html
<meta name="keywords" content="Kosal Chansothay, Sothay, Chan Sothay, KosalChansothay">
<meta name="author" content="Kosal Chansothay">
```

Keywords contain ONLY the site owner's name variants — do not add course/topic keywords (owner's explicit request).
The root `index.html` additionally has JSON-LD structured data (`WebSite` + `Person` with `alternateName` array)
mapping all name variants to `https://github.com/KosalChansothay`.

## Encoding Rule (Mojibake Prevention)

Files have been corrupted before by encoding round-trips producing `â€"` (double-encoded em dash U+2014)
and `RÃ©` (double-encoded é U+00E9). Always read/write files as UTF-8 without BOM. If mojibake appears,
fix with replacements: `â€"` -> `—`, `RÃ©` -> `Ré` (as U+00E2 U+20AC U+201D and U+00C3 U+00A9 sequences).

## Lecture HTML Contract

Each Markdown-backed lecture page must contain:

```html
<main class="main lecture-document" role="main" data-markdown="../../lecture-source.md">
```

The `data-markdown` path is relative to the lecture HTML file.

It must also contain:

```html
<div class="markdown-body" data-markdown-content aria-live="polite"></div>
```

The normal renderer fetches the Markdown file and inserts the generated HTML into this element. Existing `.card` note sections are retained as a no-JavaScript fallback and are removed after successful Markdown loading.

Load Marked.js before the shared script:

```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="../../js/script.js"></script>
```

MathJax is configured for both the original HTML delimiters and Markdown dollar delimiters:

```js
inlineMath: [['\\(', '\\)'], ['$', '$']]
displayMath: [['\\[', '\\]'], ['$$', '$$']]
```

## How To Add A New Lecture (Checklist Used For Lectures 4-6)

1. Create a Markdown file in the `notes/` folder, for example `notes/cs229-lecture-7-study-notes.md`.
2. **Strip citation references**: remove all `[123, 456]`-style transcript timestamps with the regex
   `\s*\[\d+(,\s*\d+)*\]` (verify zero remain; math vectors are unaffected).
3. Start with a clear Markdown structure and include the lecture title as the first heading.
4. Copy `courses/cs229/lecture-5.html` (the cleanest recent shell) to a new lecture HTML file.
5. Update the HTML title, description, visible `h1`, subtitle, breadcrumb, and `data-markdown` path.
6. Keep the SEO meta block (keywords = name variants only, author) in the new page.
7. Add the lecture link to **all three** index pages that maintain their own hand-written lecture lists:
   - `notes.html` (lecture index with search — use `data-note` card with `.tag` spans)
   - `courses/cs229/index.html` (course home)
   - root `index.html` (site homepage "Quick Links")
   These lists are NOT auto-generated. Missing one leaves the site inconsistent — this actually happened with
   Lecture 3 (homepage missed) and Lectures 4-5 (homepage + notes.html missed initially).
8. Keep the existing shared stylesheet and script unless the new content needs a genuinely reusable style.
9. If the lecture needs interactive charts: add `<div id="plotly-..." class="plotly-chart"></div>` containers in the
   Markdown, create a `js/lectureN-charts.js` that listens for the `markdown:rendered` event, load Plotly.js
   (`https://cdn.plot.ly/plotly-2.35.2.min.js`) and the chart script in the lecture HTML after `js/script.js`.
   See `courses/cs229/lecture-3.html` and `js/lecture3-charts.js` as the reference implementation.
10. Validate in the browser: MathJax renders (check `mjx-container` count > 0 and no raw `$$` in the DOM),
    no horizontal scroll, active nav highlighting works.

## Rendering And Local Testing

Do not open Markdown-backed lecture pages directly with `file://` when testing. Browser security blocks `fetch()` in that mode.

From the repository root, run:

```powershell
py -m http.server 8000
```

Note: use `py`, not `python` — the `python` alias is not on PATH in the current shell.

Open:

```text
http://localhost:8000/courses/cs229/lecture-1.html
http://localhost:8000/courses/cs229/lecture-2.html
```

Check that:

- Markdown headings and lists appear.
- The old fallback cards disappear after loading.
- Inline and display equations render through MathJax.
- Fenced code blocks are styled.
- The video, if present, remains responsive.
- The course navigation stays active on nested lecture pages.
- Mobile content does not overflow horizontally.
- Plotly charts (if the lecture has them) render inside their `.plotly-chart` containers after the markdown loads.

## Deployment (GitHub Pages)

- Repo: `github.com/KosalChansothay/stanford-cs229-ml-2026`
- Live site: `https://kosalchansothay.github.io/stanford-cs229-ml-2026/`
- Deploy source: `main` branch. Pushing to `main` triggers a Pages rebuild.

Expect these delays after `git push`:

1. **Pages build: 1–3 minutes** (occasionally up to 10). The live site serves the OLD version during this window.
2. **CDN/browser cache**: even after the build, the browser may show stale content. Hard refresh (Ctrl+F5) or use an incognito window to verify.

Do not assume a push failed just because the live site did not change immediately — check the repo's Actions tab for the deploy run status first. Also remember that viewing `.md` files on github.com is NOT the same as the rendered site: GitHub strips HTML (Plotly containers vanish) and does not run MathJax, so raw markdown views always look broken. Only judge rendering on the Pages site or the local server.

## Styling Rules

The lecture reading surface is intentionally document-like rather than a dashboard:

- `.lecture-document` centers the lecture blocks in a 900px reading column.
- `.markdown-body` uses a slightly larger academic reading size of `1.08rem` and `1.85` line height.
- Markdown horizontal rules receive generous vertical spacing.
- Equations and blockquotes use quiet neutral backgrounds and borders.
- Keep the palette restrained: navy, paper, gray, and subtle shadows.
- Avoid decorative colored edges, excessive gradients, oversized cards, and unrelated redesigns.

## Responsive Breakpoints (Standard System)

`css/style.css` implements a full responsive system — keep it intact:

- `<= 480px`: small phones — 15px base font, compact navbar (role hidden, name truncated), smaller headings, full-width buttons.
- `<= 768px`: phones/tablets portrait — nav collapses into Menu dropdown, tables scroll horizontally (`display: block; overflow-x: auto`).
- `769px – 1024px`: tablets/small laptops — condensed navbar spacing.
- `>= 1440px`: large desktops — `--max-width: 1160px`, 17px base font.
- `>= 1920px`: ultra-wide — `--max-width: 1240px`, 18px base font.
- `<= 480px height`: landscape phones — compact navbar.
- `prefers-reduced-motion: reduce`: disables animations/transitions.

Global guards: `body { overflow-x: hidden }`, `img/video/iframe { max-width: 100% }`, `-webkit-text-size-adjust: 100%`.
Verified at 375/768/1024/1440/1920px widths with no horizontal scroll.

## JavaScript Behavior

`js/script.js` currently does four things:

1. Computes the active sidebar link using the resolved URL.
2. Keeps the Courses link active for all `/courses/` pages.
3. Creates the mobile Menu button and toggles the sidebar navigation.
4. Loads and renders Markdown when `data-markdown` and `data-markdown-content` are present.

When changing the renderer:

- Keep the fetch path data-driven.
- Keep the HTML fallback behavior.
- Re-run MathJax after injecting Markdown.
- Do not add a framework or build system for a small feature.
- Avoid unsafe HTML processing beyond the existing Marked.js flow unless sanitization is deliberately added.

## Known Constraints

- Marked.js and MathJax are loaded from CDNs, so offline rendering is not guaranteed.
- YouTube embedding can fail in `file://` previews. Lecture 1 has a direct YouTube fallback link.
- The repository has no automated test suite or package configuration.
- The two lecture HTML files still contain duplicated fallback note bodies. Treat them as fallback content, not the primary source.
- Keep changes focused and do not revert user edits in unrelated files.

## Recommended Continuation Prompt

When asking another model to continue, provide this file first and say:

> Read `AI_CONTEXT.md` first. Work from the Markdown source of truth, preserve the static academicpages-inspired architecture, make the smallest focused change, and validate the affected lecture page over HTTP.
