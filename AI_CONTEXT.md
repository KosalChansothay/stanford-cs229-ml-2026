# AI Context: Machine Learning Note

## Purpose

This repository is a small static academic notes website for Stanford CS229 machine learning lectures. It is inspired by the academicpages GitHub template. The site is intentionally simple: HTML, CSS, JavaScript, Markdown, MathJax, and CDN-loaded Marked.js. There is no npm project, bundler, framework, or build step.

Read this file before changing the project. Prefer the existing patterns and make focused edits.

## Current Source Of Truth

Lecture content is maintained in Markdown files in the `notes/` folder:

- `notes/cs229-lecture-1-study-notes.md` -> `courses/cs229/lecture-1.html`
- `notes/cs229-lecture-2-study-notes.md` -> `courses/cs229/lecture-2.html`
- `notes/cs229-lecture-3-study-notes.md` -> `courses/cs229/lecture-3.html`

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

Shared assets:

- `css/style.css`: all layout, typography, responsive, academic document, and component styles (including `.plotly-chart` containers)
- `js/script.js`: active navigation, mobile menu, lecture filtering, Markdown loading, MathJax re-typesetting, and the `markdown:rendered` event
- `js/lecture3-charts.js`: page-specific Plotly charts for Lecture 3, drawn after `markdown:rendered` fires

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

## How To Add A New Lecture

1. Create a Markdown file in the `notes/` folder, for example `notes/cs229-lecture-4-study-notes.md`.
2. Start with a clear Markdown structure and include the lecture title as the first heading.
3. Copy `courses/cs229/lecture-2.html` to a new lecture HTML file.
4. Update the HTML title, description, visible `h1`, subtitle, breadcrumb, and `data-markdown` path.
5. Remove or update the video block if the new lecture has a different video.
6. Add the lecture link to `notes.html` and `courses/cs229/index.html`.
7. Keep the existing shared stylesheet and script unless the new content needs a genuinely reusable style.
8. If the lecture needs interactive charts: add `<div id="plotly-..." class="plotly-chart"></div>` containers in the Markdown, create a `js/lectureN-charts.js` that listens for the `markdown:rendered` event, load Plotly.js (`https://cdn.plot.ly/plotly-2.35.2.min.js`) and the chart script in the lecture HTML after `js/script.js`. See `courses/cs229/lecture-3.html` and `js/lecture3-charts.js` as the reference implementation.

## Rendering And Local Testing

Do not open Markdown-backed lecture pages directly with `file://` when testing. Browser security blocks `fetch()` in that mode.

From the repository root, run:

```powershell
python -m http.server 8000
```

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

## Styling Rules

The lecture reading surface is intentionally document-like rather than a dashboard:

- `.lecture-document` centers the lecture blocks in a 900px reading column.
- `.markdown-body` uses a slightly larger academic reading size of `1.08rem` and `1.85` line height.
- Markdown horizontal rules receive generous vertical spacing.
- Equations and blockquotes use quiet neutral backgrounds and borders.
- Keep the palette restrained: navy, paper, gray, and subtle shadows.
- Avoid decorative colored edges, excessive gradients, oversized cards, and unrelated redesigns.
- Preserve mobile behavior at `768px` and below.

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
