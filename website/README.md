# ReasonCLIP project page

Static page (no build step). Preview locally:

```bash
python -m http.server 8765 --directory website
```

- `index.html`, `style.css`, `app.js`: page, monochrome styling shared with MLLM Gallery, interactions.
- `data/results.js`: numbers transcribed from the paper (Tables 3–6, 17). Edit here to update charts and tables.
- `data/bench.js` + `assets/bench/`: 8 examples from `RISys-Lab/RCLIP-Bench` (DOCCI images, resized to 1000px).
- `assets/method.jpg`: downscaled copy of `doc/method.png`.

Bump the `?v=` query on the CSS/JS links in `index.html` after edits so browsers pick up the new files.
