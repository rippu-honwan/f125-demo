// Three.js scene for the 3D Learn mode: track ribbon, car, corner markers, cameras.
// Imported dynamically by index.js, so a CDN/WebGL failure only disables the canvas.
import * as THREE from "three";
import { bisect, cuePoint, lineHeadings } from "./core.js";

// ponytail: fixed 14 m track; real widths, kerbs and run-off aren't in the data.
const TRACK_HALF_WIDTH = 7;
const CAMS = {
  tcam: { back: 0.6, height: 1.7, ahead: 40, look: 1.0 },   // above the airbox
  chase: { back: 9, height: 3.2, ahead: 8, look: 0.5 },
};

/**
 * Builds the scene into `container`. Throws if WebGL is unavailable or the data is unusable.
 * `cars`: one {color, lane} per car; lane = sideways offset (m) so cars at the same spot don't overlap.
 * `youLine`: your lap's points (3D Compare with your own line) for a second line ribbon, else null.
 */
export function createScene(container, points, corners, cars, youLine = null) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  try {
    return buildScene(renderer, container, points, corners, cars, youLine);
  } catch (err) {
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();
    throw err;
  }
}

/** Everything after the renderer exists; split out so a failure can release the GL context. */
function buildScene(renderer, container, points, corners, cars, youLine) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9cc3dd);
  scene.fog = new THREE.Fog(0x9cc3dd, 400, 2500);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x556644, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(300, 600, 200);
  scene.add(sun);

  // world (x, y, elev) -> three (x, elev, -y), centred so floats stay small
  const n = points.x.length;
  const cx = (Math.min(...points.x) + Math.max(...points.x)) / 2;
  const cy = (Math.min(...points.y) + Math.max(...points.y)) / 2;
  const zMin = Math.min(...points.z);
  const toThree = (x, y, z, out = new THREE.Vector3()) => out.set(x - cx, z - zMin, -(y - cy));
  const path = (p) => Array.from({ length: p.x.length }, (_, i) => toThree(p.x[i], p.y[i], p.z[i]));
  const pos = path(points);
  const up = new THREE.Vector3(0, 1, 0);
  // Smoothed line headings, not neighbour differences: those flip where positions stall or step back,
  // which twisted the road into dark slivers (Suzuka's start/finish, Jeddah at 4.2 km).
  const heading = points.heading || lineHeadings(points);

  // A strip along a path (three.js points + headings), offset sideways by ±half; the last quad closes the loop.
  const ribbon = (pts, hd, half, lift) => {
    const m = pts.length;
    const verts = new Float32Array(m * 6);
    const side = new THREE.Vector3();
    for (let i = 0; i < m; i++) {
      side.set(Math.sin(hd[i]), 0, Math.cos(hd[i])).multiplyScalar(half);   // right of travel
      const p = pts[i];
      verts.set([p.x + side.x, p.y + lift, p.z + side.z, p.x - side.x, p.y + lift, p.z - side.z], i * 6);
    }
    const index = [];
    for (let i = 0; i < m; i++) {
      const a = 2 * i, b = a + 1, c = (2 * (i + 1)) % (2 * m), d = c + 1;
      index.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    g.setIndex(index);
    return g;
  };
  const road = ribbon(pos, heading, TRACK_HALF_WIDTH, 0.05);
  // Where positions step back against d the ribbon folds over itself: those quads run backwards, so
  // they face down. Front faces only (the road faces up) drops them; forward quads already cover the
  // spot. Normals point straight up, as computed ones would average up with down into dark seams.
  // ponytail: slopes shade like flat road.
  road.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(n * 6).map((_, k) => (k % 3 === 1 ? 1 : 0)), 3));
  scene.add(new THREE.Mesh(road, new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.9 })));

  // Racing line for Drive: the pro's line, red where braking, green at full throttle, yellow between.
  const lineGeo = ribbon(pos, heading, 0.35, 0.12);
  const cols = new Float32Array(n * 6), col = new THREE.Color();
  for (let i = 0; i < n; i++) {
    col.set(points.brake[i] >= 0.5 ? 0xff2e2e : points.throttle[i] >= 0.95 ? 0x3ddc84 : 0xffd400);
    cols.set([col.r, col.g, col.b, col.r, col.g, col.b], i * 6);
  }
  lineGeo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
  const line = new THREE.Mesh(lineGeo, new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2,
  }));
  line.visible = false;
  scene.add(line);

  // 3D Compare: your line in your car's colour, shown and hidden with the pro's.
  let youRibbon = null;
  if (youLine) {
    youRibbon = new THREE.Mesh(ribbon(path(youLine), youLine.heading || lineHeadings(youLine), 0.35, 0.13),
      new THREE.MeshBasicMaterial({ color: cars[1].color, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3 }));
    youRibbon.visible = false;
    scene.add(youRibbon);
  }

  // ponytail: flat ground at the lowest point; hilly tracks show the road raised above it.
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000),
    new THREE.MeshStandardMaterial({ color: 0x5d7a4a, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -1.5;
  scene.add(ground);

  // A thin box lying across the road at lap distance d.
  const stripe = (d, color, depth) => {
    const i = bisect(points.d, d);
    const m = new THREE.Mesh(new THREE.BoxGeometry(depth, 0.08, TRACK_HALF_WIDTH * 2),
      new THREE.MeshBasicMaterial({ color }));
    m.position.copy(pos[i]).y += 0.1;
    m.rotation.y = heading[i];
    scene.add(m);
    return m;
  };
  stripe(points.d[0], 0xffffff, 1.5);                         // start / finish
  const cues = [];                                            // brake/lift stripes; hidden in Hard practice
  for (const c of corners) {
    if (c.action === "brake") cues.push(stripe(cuePoint(c), 0xff2e2e, 2));
    else if (c.action === "lift") cues.push(stripe(cuePoint(c), 0xff9900, 1.2));
    const apex = new THREE.Mesh(new THREE.SphereGeometry(1.2, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd400 }));
    apex.position.copy(pos[bisect(points.d, c.apex_m)]).y += 1.2;
    scene.add(apex);
  }

  const meshes = cars.map((c) => {
    const m = buildCar(c.color);
    scene.add(m);
    return m;
  });

  const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 3500);
  const resize = () => {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  container.prepend(renderer.domElement);
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  const fwd = new THREE.Vector3(), target = new THREE.Vector3();

  /**
   * Place car k at `states[k]` ({x, y, z, heading}, e.g. from core.stateAt), aim camera `cam` at car
   * `follow`, render one frame. A car with no state (missing or null) is hidden.
   */
  function update(states, cam, follow = 0) {
    meshes.forEach((m, k) => {
      const s = states[k];
      m.visible = !!s;
      if (!s) return;
      toThree(s.x, s.y, s.z, m.position).y += 0.05;
      const lane = cars[k].lane || 0;                          // sideways, perpendicular to the heading
      m.position.x += Math.sin(s.heading) * lane;
      m.position.z += Math.cos(s.heading) * lane;
      m.rotation.y = s.heading;
    });
    const s = states[follow];
    fwd.set(Math.cos(s.heading), 0, -Math.sin(s.heading));
    const p = meshes[follow].position;
    if (cam === "top") {
      camera.up.copy(fwd);                                     // heading-up overhead view
      camera.position.set(p.x, p.y + 75, p.z);
      camera.lookAt(p);
    } else {
      const c = CAMS[cam] || CAMS.tcam;
      camera.up.copy(up);
      camera.position.copy(p).addScaledVector(fwd, -c.back);
      camera.position.y += c.height;
      target.copy(p).addScaledVector(fwd, c.ahead);
      target.y += c.look;
      camera.lookAt(target);
    }
    renderer.render(scene, camera);
  }

  function dispose() {
    ro.disconnect();
    scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    renderer.dispose();
    renderer.forceContextLoss();   // browsers cap live WebGL contexts (~16)
    renderer.domElement.remove();
  }

  /** Show or hide the brake/lift stripes (the start/finish line stays). */
  function setCues(on) {
    for (const m of cues) m.visible = on;
  }

  const ghosted = cars.map(() => false);
  /** Draw car k see-through (Drive's ghost) or solid. */
  function setGhost(k, on) {
    if (ghosted[k] === on) return;
    ghosted[k] = on;
    meshes[k].traverse((o) => {
      if (!o.material) return;
      o.material.transparent = on;
      o.material.opacity = on ? 0.35 : 1;
      o.material.depthWrite = !on;
      o.material.needsUpdate = true;
    });
  }

  /** Show or hide the racing line(s): the pro's, and yours in 3D Compare. */
  function setLine(on) {
    line.visible = on;
    if (youRibbon) youRibbon.visible = on;
  }

  return { update, dispose, setCues, setGhost, setLine };
}

/** A generic open-wheel silhouette from primitives — original, no real livery or branding. Nose = +x. */
function buildCar(color) {
  const g = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.2 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x15161a, roughness: 0.8 });
  const add = (geo, mat, x, y, z) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    g.add(m);
    return m;
  };
  add(new THREE.BoxGeometry(4.6, 0.45, 0.9), paint, 0, 0.45, 0);      // tub + nose
  add(new THREE.BoxGeometry(1.6, 0.35, 1.5), paint, -0.6, 0.4, 0);    // sidepods
  add(new THREE.BoxGeometry(0.9, 0.5, 0.5), paint, -0.9, 0.9, 0);     // engine cover
  add(new THREE.BoxGeometry(0.3, 0.06, 1.9), paint, 2.5, 0.25, 0);    // front wing
  add(new THREE.BoxGeometry(0.35, 0.08, 1.1), dark, -2.3, 1.0, 0);    // rear wing
  for (const z of [0.5, -0.5]) add(new THREE.BoxGeometry(0.4, 0.6, 0.05), dark, -2.3, 0.75, z);  // wing pillars
  for (const [x, z] of [[1.7, 0.85], [1.7, -0.85], [-1.6, 0.85], [-1.6, -0.85]]) {
    add(new THREE.CylinderGeometry(0.36, 0.36, 0.4, 16), dark, x, 0.36, z).rotation.x = Math.PI / 2;
  }
  return g;
}
