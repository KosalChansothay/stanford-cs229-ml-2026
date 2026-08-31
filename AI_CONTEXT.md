# AI Context: Machine Learning Note

## Purpose

This repository is a small static academic notes website for Stanford CS229 machine learning lectures. It is inspired by the academicpages GitHub template. The site is intentionally simple: HTML, CSS, JavaScript, Markdown, MathJax, and CDN-loaded Marked.js. There is no npm project, bundler, framework, or build step.

Read this file before changing the project. Prefer the existing patterns and make focused edits.

## Current Source Of Truth

### CS229: Foundations and Paradigms of Machine Learning

- `notes/cs229-lecture-1-study-notes.md` -> `courses/cs229/lecture-1.html`
- `notes/cs229-lecture-2-study-notes.md` -> `courses/cs229/lecture-2.html`
- `notes/cs229-lecture-3-study-notes.md` -> `courses/cs229/lecture-3.html`
- `notes/cs229-lecture-4-study-notes.md` -> `courses/cs229/lecture-4.html`
- `notes/cs229-lecture-5-study-notes.md` -> `courses/cs229/lecture-5.html`
- `notes/cs229-lecture-6-study-notes.md` -> `courses/cs229/lecture-6.html`
- `notes/cs229-lecture-7-study-notes.md` -> `courses/cs229/lecture-7.html`
- `notes/cs229-lecture-8-study-notes.md` -> `courses/cs229/lecture-8.html`
- `notes/cs229-lecture-9-study-notes.md` -> `courses/cs229/lecture-9.html`
- `notes/cs229-lecture-10-study-notes.md` -> `courses/cs229/lecture-10.html`
- `notes/cs229-lecture-11-study-notes.md` -> `courses/cs229/lecture-11.html`
- `notes/cs229-lecture-12-study-notes.md` -> `courses/cs229/lecture-12.html`
- `notes/cs229-lecture-13-study-notes.md` -> `courses/cs229/lecture-13.html`
- `notes/cs229-lecture-14-study-notes.md` -> `courses/cs229/lecture-14.html`

### CS336: Language Modeling from Scratch

- `notes/cs336-lecture-1-study-notes.md` -> `courses/cs336/lecture-1.html`
- `notes/cs336-lecture-2-notes.md` -> `courses/cs336/lecture-2.html`
- `notes/cs336-lecture-3-notes.md` -> `courses/cs336/lecture-3.html`
- `notes/cs336-lecture-4-notes.md` -> `courses/cs336/lecture-4.html`
- `notes/cs336-lecture-5-notes.md` -> `courses/cs336/lecture-5.html`
- `notes/cs336-lecture-6-notes.md` -> `courses/cs336/lecture-6.html`
- `notes/cs336-lecture-7-notes.md` -> `courses/cs336/lecture-7.html`
- `notes/cs336-lecture-8-notes.md` -> `courses/cs336/lecture-8.html`
- `notes/cs336-lecture-9-notes.md` -> `courses/cs336/lecture-9.html`
- `notes/cs336-lecture-10-notes.md` -> `courses/cs336/lecture-10.html`
- `notes/cs336-lecture-11-notes.md` -> `courses/cs336/lecture-11.html`
- `notes/cs336-lecture-12-notes.md` -> `courses/cs336/lecture-12.html`
- `notes/cs336-lecture-13-notes.md` -> `courses/cs336/lecture-13.html`
- `notes/cs336-lecture-14-notes.md` -> `courses/cs336/lecture-14.html`
- `notes/cs336-lecture-15-notes.md` -> `courses/cs336/lecture-15.html`
- `notes/cs336-lecture-16-notes.md` -> `courses/cs336/lecture-16.html`
- `notes/cs336-lecture-17-notes.md` -> `courses/cs336/lecture-17.html`
- `notes/cs336-lecture-guest-notes.md` -> `courses/cs336/lecture-guest.html`

Lecture Markdown files must NOT contain bracketed numeric citation references such as `[358, 429]` or `[364]`.
They are leftover transcript timestamps and are stripped before publishing. When adding new lecture notes,
remove them with a regex like `\s*\[\d+(,\s*\d+)*\]` (this safely preserves math like one-hot vectors `[1, 0, 0, 0]`
because those contain zeros and live inside `$...$`).

**WARNING — the regex has a known failure mode**: it EATS one-hot vectors like `[1, 0, 0, 0]` when they appear
as bare text (0 is a digit). This corrupted Lecture 4's one-hot examples. After stripping, always grep for
artifacts like `=\^T` and restore any eaten vectors manually.

Edit the Markdown files for lecture content. Do not copy new lecture content into the HTML pages unless a fallback is intentionally being updated.

Markdown supports:

- ATX headings: `## Section`, `### Subsection`
- Unordered and ordered lists
- Fenced code blocks
- Inline math: `$x$`
- Display math: `$$ ... $$`
- Horizontal rules: `---`
- Plotly chart containers: `<div id="plotly-..." class="plotly-chart"></div>` rendered by page-specific scripts in `js/` (see `js/cs229-lecture3-charts.js` for the pattern)

Keep the Markdown readable and structured like academic notes: summary, definitions, derivations, algorithms or workflows, examples, reflection questions, and references.

## Page Architecture

Root pages:

- `index.html`: site home and featured lectures
- `about.html`: project description and technology information
- `notes.html`: lecture index with client-side search

Course pages:

- `courses/index.html`: all courses
- `courses/cs229/index.html`: CS229 course home (Lectures 1–14)
- `courses/cs229/lecture-1.html` through `lecture-14.html`: CS229 lecture shells and Markdown renderers
- `courses/cs336/index.html`: CS336 course home (Lectures 1–17 + Guest)
- `courses/cs336/lecture-1.html` through `lecture-17.html` & `lecture-guest.html`: CS336 lecture shells and Markdown renderers

Shared assets:

- `css/style.css`: all layout, typography, responsive, academic document, component styles, dark mode tokens (Warm Academic Charcoal), code syntax highlighting, and `.plotly-chart` containers
- `js/script.js`: active navigation, dark mode match/toggle with localStorage, mobile menu, lecture filtering, Markdown loading, code copy buttons, MathJax re-typesetting, and the `markdown:rendered` event
- `js/cs229-lecture1-charts.js` through `js/cs229-lecture14-charts.js`: Plotly charts for CS229 lectures
- `js/cs336-lecture1-charts.js` through `js/cs336-lecture17-charts.js` & `js/cs336-lectureguest-charts.js`: Plotly charts for CS336 lectures (Roofline, GQA, MoE routing, FlashAttention IO, Triton kernels, Ring AllReduce, ZeRO memory, Scaling laws, Speculative decoding, muP transfer, Elo curves, Data waterfalls, LSH S-curves, DPO margins, GRPO rollouts, SigLIP losses, Disaggregated serving, and Mega-kernels)
- `img/profile.jpg`: site logo/avatar used in the top navigation bar on every page

## Dark Mode System (Warm Academic Charcoal)

The site includes a comprehensive "Match System / Toggle" dark mode inspired by GitHub Academic Pages:

- **Theme Palette**:
  - Canvas / Background: `#121517` with subtle top gradient.
  - Cards & Surfaces: `#1a1f23` elevated with subtle `#282e33` borders.
  - Typography: `#f0f3f6` (headings), `#e6edf3` (subheadings), `#cbd5dc` (body text), `#8e9aa2` (muted text), `#6cb6ff` (primary accents).
- **Match System & Zero-FOUC**: An immediate anti-FOUC script in `js/script.js` checks `localStorage` and system `prefers-color-scheme: dark` before render.
- **Manual Toggle**: Sun/moon toggle button (`.theme-toggle-btn`) in the sticky top navigation.
- **Instant Transitions**: Theme changes and page navigation occur instantly (`--theme-transition: none;`) to eliminate cross-page color morphing or lag.
- **Syntax Highlighting**: High-contrast One Dark / Tokyo Night token coloring (`#e6edf3` variables/identifiers, `#ff7b72` keywords, `#79c0ff` functions, `#ffa657` classes, `#f0883e` self, `#7ee787` strings, `#8b949e` comments).

## Interactive Charts Policy (Plotly Placeholders)

Lecture Markdown may contain diagram placeholders of the form:

```text
[Insert diagram: <description of the desired visualization>]
```

**When you encounter one, do NOT leave it as plain text.** Replace it with an interactive Plotly chart —
static descriptions waste the opportunity to build intuition. The established workflow:

1. Replace the placeholder block in the Markdown with a container:
   ```html
   <div id="plotly-<lecture>-<name>" class="plotly-chart" aria-label="Interactive Plotly chart: <description>"></div>
   ```
   Optionally follow it with a one-line `<p><em>Figure: ...</em></p>` caption.
2. Create (or extend) `js/cs<course>-lectureN-charts.js` following the pattern in `js/cs229-lecture3-charts.js` and `js/cs336-lecture6-charts.js`:
   - Wrap everything in an IIFE; use a seeded PRNG (`mulberry32`) so synthetic data is stable across reloads.
   - Compute fits/trajectories mathematically (closed-form least squares, gradient steps) rather than hard-coding decorative curves.
   - **Chart Background**: Charts use a clean white card background (`#ffffff` canvas and SVG `.bg`) for consistent, high-contrast visualization in both light and dark modes.
   - **Log-Scale Axes**: When `type: 'log'` is used, always provide explicit `tickmode: 'array'`, `tickvals`, and `ticktext` (e.g. `[64, 128, 256, 512, 1024, ...]` or `['10 MB', '100 MB', '1 GB', ...]`) to prevent Plotly's default sub-decade single-digit mantissa artifacts (`2, 3, 4, 5...`).
   - **Breathing Room & Margins**:
     - Dual-axis charts (`yaxis2`): `margin: { t: 90, b: 55, l: 75, r: 105 }` to ensure right-axis titles and tick labels have ample clearance.
     - Single-axis charts: `margin: { t: 90, b: 55, l: 70, r: 50 }`.
     - Title position: `y: 0.98`, horizontal legend subtitle: `y: 1.16`.
   - Export `window.renderCs<course>LectureNCharts` that draws each chart by container id.
   - Listen for the `markdown:rendered` event (charts live inside the fetched markdown), with a `readyState` guard fallback.
3. Load Plotly and the chart script in the lecture HTML after `js/script.js`:
   ```html
   <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
   <script src="../../js/cs336-lectureN-charts.js"></script>
   ```
4. Validate in the browser: container exists, `.main-svg` is present (chart actually rendered), hover works,
   and the layout stays responsive.

## Plotly Slider Pitfalls (Learned in Lectures 9-10)

Slider-driven charts have two failure modes that cost debugging time — avoid them:

1. **Trace-count stability**: every slider step MUST produce the same number of traces. If a trace
   (e.g., an EM trajectory) only exists at later steps, emit it as an EMPTY trace (`x: [], y: []`) at
   step 0. Mismatched counts corrupt `gd.data` and silently break all subsequent updates.
2. **Slider `update`/`restyle` methods may not apply** in some environments even with correct args.
   Robust pattern: store the correct restyle args in the steps AND add a backup listener:
   ```js
   gd.on('plotly_sliderupdate', function (ev) {
       var step = gd.layout.sliders[0].steps[ev.slider.active];
       if (step && step.args) Plotly.restyle(gd, step.args[0], step.args[1]);
   });
   ```
   To verify programmatically: `Plotly.restyle(gd, step.args[0], step.args[1])` manually — if the data
   moves, the args are correct and real user clicks will work.
3. **marked.js + math escaping**: strip citations AND escape `_{` → `\_{` in the same pass. Two unescaped
   `_{...}` in one paragraph pair as `<em>` and break adjacent `$$` blocks.

## Navigation Layout & Academic Underline

The navigation is a **sticky top navbar** (`.sidebar` class kept for historical reasons), not a left sidebar:

- `position: sticky; top: 0` horizontal flex bar: logo image + site name on the left, nav links right-aligned, theme toggle button, and GitHub icon at the end.
- **Academic Underline**: Active navigation links use pure serif typography (`font-weight: 600`) with a crisp 2px bottom accent rule (`border-bottom: 2px solid var(--primary); background: transparent; box-shadow: none;`), avoiding artificial 3D button boxes.
- The avatar is `<img class="sidebar-avatar" src=".../img/profile.jpg">` (44px circle, `object-fit: cover`), NOT an emoji div.
- `.sidebar-bio` and `.sidebar-footer` are hidden via CSS.
- `body` must NOT have `display: flex` — that stretches the navbar to full page height.
- `.main` is centered with `max-width: var(--max-width)` and no left margin.
- On `<= 768px` the nav collapses into a "Menu" dropdown toggle (created by `js/script.js`).

## Lecture Pagination (Previous / Next Navigation)

Every lecture HTML page contains standardized `.lecture-pagination` controls:
- Provides dual directional navigation buttons with topic subtitles and arrows (`&larr;` / `&rarr;`).
- First lectures route back to course overviews (`courses/cs336/index.html` or `courses/cs229/index.html`), intermediate lectures navigate sequentially, and final lectures link back to syllabus indices.
- Fully responsive, collapsing into stacked cards on mobile devices.

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
   See `courses/cs229/lecture-3.html` and `js/cs229-lecture3-charts.js` as the reference implementation.
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
