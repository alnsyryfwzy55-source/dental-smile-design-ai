"""Builds the app into ../docs (served by GitHub Pages). The AI runtimes live in ../docs/vendor; put teeth_unet_v2.onnx in ../docs."""
import pathlib, shutil, time
R = pathlib.Path(__file__).parent; S = R / 'src'; D = R.parent / 'docs'
D.mkdir(exist_ok=True)
shutil.copy(S / 'index.template.html', D / 'index.html')
for f in ['selftest.jpg', 'selftest.json', 'style.css', 'core.js', 'smile.js', 'sim.js', 'library.js', 'engine.js', 'app.js', 'capture.js', 'results.js', 'settings.js', 'worker.js', 'manifest.webmanifest', 'icon.svg']:
    shutil.copy(S / f, D / f)
(D / 'sw.js').write_text((S / 'sw.js').read_text().replace('__BUILD__', time.strftime('%Y%m%d%H%M%S')))
missing = [p for p in ['vendor/ort.webgpu.min.js', 'vendor/face_landmarker.task', 'vendor/mediapipe/vision_bundle.mjs', 'teeth_unet_v2.onnx'] if not (D / p).exists()]
print('built ../docs' + (' | missing: ' + ', '.join(missing) if missing else ''))
