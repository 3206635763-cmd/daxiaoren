'use client';

import {
  type CSSProperties,
  type ChangeEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  Camera,
  CameraOff,
  ArrowLeft,
  ChevronRight,
  Dumbbell,
  ImagePlus,
  Settings2,
  Upload,
  Volume2,
  VolumeX,
} from 'lucide-react';
import type {
  BoundingBox,
  FaceDetector as FaceDetectorInstance,
  ImageSegmenter as ImageSegmenterInstance,
  PoseLandmarker as PoseLandmarkerInstance,
} from '@mediapipe/tasks-vision';
import type * as ThreeTypes from 'three';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';

type HitDirection = 'left' | 'right';
type CameraState = 'idle' | 'loading' | 'ready' | 'error';
type GameMode = 'standard' | 'practice';
type GamePhase = 'menu' | 'intro' | 'playing' | 'ended';
type IntroStep = 'waiting' | 'ready' | 'fight';
type Landmark = { x: number; y: number; z: number; visibility?: number };
type HandSample = { x: number; y: number; z: number; extension: number; visible: number };
type PoseSample = { at: number; left: HandSample; right: HandSample };
type PunchTracker = {
  history: PoseSample[];
  lastPunchAt: number;
};
type MannequinRig = {
  root: ThreeTypes.Group;
  torso: ThreeTypes.Group;
  head: ThreeTypes.Group;
  leftArm: ThreeTypes.Group;
  rightArm: ThreeTypes.Group;
};
type SpringMotion = { position: number; velocity: number };
type HitMotion = {
  headRoll: SpringMotion;
  headYaw: SpringMotion;
  headShift: SpringMotion;
  torsoRoll: SpringMotion;
  torsoYaw: SpringMotion;
  torsoShift: SpringMotion;
  depth: SpringMotion;
};
type DamageSurface = {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  texture: ThreeTypes.CanvasTexture;
};
type DamageSystem = { face: DamageSurface; body: DamageSurface };

let THREE: typeof import('three');

const particleColors = ['#ff5b35', '#ffb02e', '#67e8f9', '#ffffff'];
const impactWords = ['砰!', '重击!', '漂亮!', '继续!', '痛快!', '破防!', '暴击!', '全开!'];
const roundSeconds = 30;
const targetPunches = 50;
const readyClipDurationMs = 3631;
const readyFightAtMs = 1850;
const punchSoundSources = ['/audio/punch-1.mp3', '/audio/punch-2.mp3', '/audio/punch-3.mp3', '/audio/punch-4.mp3'];
const arenaBackgrounds = [
  { id: 'dungeon', name: '地下刑房', src: '/backgrounds/dungeon.jpeg' },
  { id: 'interrogation', name: '审讯室', src: '/backgrounds/interrogation.jpeg' },
  { id: 'arena', name: '格斗赛场', src: '/backgrounds/arena.jpeg' },
] as const;

function pointDistance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function createPunchTracker(): PunchTracker {
  return { history: [], lastPunchAt: 0 };
}

function findPunch(landmarks: Landmark[], now: number, state: PunchTracker): HitDirection | null {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftElbow = landmarks[13];
  const rightElbow = landmarks[14];
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];
  if (!leftShoulder || !rightShoulder || !leftElbow || !rightElbow || !leftWrist || !rightWrist) return null;

  const shoulderWidth = pointDistance(leftShoulder, rightShoulder);
  const shouldersVisible = Math.min(leftShoulder.visibility ?? 1, rightShoulder.visibility ?? 1) > 0.42;
  if (!shouldersVisible || shoulderWidth < 0.055) return null;

  const normalizeHand = (wrist: Landmark, shoulder: Landmark, elbow: Landmark): HandSample => {
    const x = (wrist.x - shoulder.x) / shoulderWidth;
    const y = (wrist.y - shoulder.y) / shoulderWidth;
    return {
      x,
      y,
      z: (wrist.z - shoulder.z) / shoulderWidth,
      extension: Math.hypot(x, y),
      visible: Math.min(wrist.visibility ?? 1, elbow.visibility ?? 1),
    };
  };

  const sample: PoseSample = {
    at: now,
    left: normalizeHand(leftWrist, leftShoulder, leftElbow),
    right: normalizeHand(rightWrist, rightShoulder, rightElbow),
  };
  state.history.push(sample);
  state.history = state.history.filter((item) => now - item.at <= 220);

  if (now - state.lastPunchAt < 210) return null;
  const baseline = state.history.find((item) => now - item.at >= 75 && now - item.at <= 170);
  if (!baseline) return null;

  const candidates: Array<{ hand: HitDirection; score: number }> = [];
  (['left', 'right'] as const).forEach((hand) => {
    const current = sample[hand];
    const before = baseline[hand];
    const dt = Math.max((now - baseline.at) / 1000, 0.07);
    const planarTravel = Math.hypot(current.x - before.x, current.y - before.y);
    const forwardTravel = before.z - current.z;
    const extensionGain = current.extension - before.extension;
    const travel = Math.hypot(planarTravel, Math.max(0, forwardTravel) * 0.85);
    const speed = travel / dt;
    const qualifies =
      current.visible > 0.42 &&
      before.visible > 0.4 &&
      current.extension > 0.72 &&
      travel > 0.11 &&
      speed > 0.9 &&
      (extensionGain > 0.045 || forwardTravel > 0.025);
    if (qualifies) candidates.push({ hand, score: speed + Math.max(0, forwardTravel) * 2.5 });
  });

  if (!candidates.length) return null;
  candidates.sort((a, b) => b.score - a.score);
  const winner = candidates[0].hand;
  state.lastPunchAt = now;
  state.history = [sample];
  return winner;
}

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const x = clamp01((value - edge0) / (edge1 - edge0));
  return x * x * (3 - 2 * x);
}

function createHitMotion(): HitMotion {
  const channel = (): SpringMotion => ({ position: 0, velocity: 0 });
  return {
    headRoll: channel(),
    headYaw: channel(),
    headShift: channel(),
    torsoRoll: channel(),
    torsoYaw: channel(),
    torsoShift: channel(),
    depth: channel(),
  };
}

function stepSpring(channel: SpringMotion, stiffness: number, damping: number, delta: number) {
  const acceleration = -stiffness * channel.position - damping * channel.velocity;
  channel.velocity += acceleration * delta;
  channel.position += channel.velocity * delta;
}

function createNormalizedPhoto(image: ImageBitmap) {
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 768 / Math.max(image.width, image.height));
  canvas.width = Math.max(1, Math.round(image.width * scale));
  canvas.height = Math.max(1, Math.round(image.height * scale));
  const context = canvas.getContext('2d');
  if (!context) throw new Error('canvas unavailable');
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function maskValue(mask: Float32Array, width: number, height: number, x: number, y: number) {
  const maskX = Math.min(width - 1, Math.max(0, Math.floor(x * width)));
  const maskY = Math.min(height - 1, Math.max(0, Math.floor(y * height)));
  return mask[maskY * width + maskX] ?? 0;
}

function createHeadTexture(
  image: HTMLCanvasElement,
  box?: BoundingBox,
  personMask?: { data: Float32Array; width: number; height: number },
) {
  const detectedWidth = box?.width ?? Math.min(image.width * 0.64, image.height * 0.64);
  const detectedHeight = box?.height ?? detectedWidth * 1.12;
  const centerX = box ? box.originX + box.width / 2 : image.width / 2;
  const centerY = box ? box.originY + box.height * 0.44 : image.height * 0.35;
  const cropWidth = Math.min(image.width, detectedWidth * (box ? 1.36 : 1.2));
  const cropHeight = Math.min(image.height, detectedHeight * (box ? 1.48 : 1.28));
  const cropX = Math.max(0, Math.min(image.width - cropWidth, centerX - cropWidth / 2));
  const cropY = Math.max(0, Math.min(image.height - cropHeight, centerY - cropHeight / 2));
  const canvas = document.createElement('canvas');
  canvas.width = 420;
  canvas.height = 500;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('canvas unavailable');
  context.drawImage(image, cropX, cropY, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const sourceX = (cropX + (x / canvas.width) * cropWidth) / image.width;
      const sourceY = (cropY + (y / canvas.height) * cropHeight) / image.height;
      const person = personMask
        ? smoothstep(0.2, 0.68, maskValue(personMask.data, personMask.width, personMask.height, sourceX, sourceY))
        : 1;
      const normalizedX = (x / canvas.width - 0.5) / 0.515;
      const normalizedY = (y / canvas.height - 0.47) / 0.525;
      const headShape = Math.pow(Math.abs(normalizedX), 2.7) + Math.pow(Math.abs(normalizedY), 2.3);
      const vertical = y / canvas.height;
      const neckProgress = smoothstep(0.76, 0.98, vertical);
      const neckHalfWidth = 0.28 - neckProgress * 0.13;
      const neckCut = vertical < 0.76
        ? 1
        : smoothstep(neckHalfWidth + 0.045, neckHalfWidth, Math.abs(x / canvas.width - 0.5));
      const bottomFade = smoothstep(1, 0.93, vertical);
      const shape = smoothstep(1.08, 0.95, headShape) * neckCut * bottomFade;
      const alphaIndex = (y * canvas.width + x) * 4 + 3;
      pixels.data[alphaIndex] = Math.round(pixels.data[alphaIndex] * person * shape);
    }
  }
  context.putImageData(pixels, 0, 0);
  return canvas;
}

function createBodyMarkTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 520;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('canvas unavailable');
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = '#242220';
  context.font = '900 154px "PingFang SC", "Microsoft YaHei", sans-serif';
  context.fillText('有', canvas.width / 2, 145);
  context.fillText('罪', canvas.width / 2, 325);
  return canvas;
}

function createDamageSurface(width: number, height: number): DamageSurface {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('canvas unavailable');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return { canvas, context, texture };
}

function paintDamage(surface: DamageSurface, hitIndex: number) {
  const { canvas, context, texture } = surface;
  const severity = Math.min(0.82, 0.1 + Math.log2(hitIndex + 1) * 0.075);
  const x = canvas.width * (0.17 + Math.random() * 0.66);
  const y = canvas.height * (0.14 + Math.random() * 0.72);
  const radius = canvas.width * (0.025 + Math.random() * 0.055) * (0.88 + severity * 0.34);
  const angle = (Math.random() - 0.5) * Math.PI;
  const markType = Math.floor(Math.random() * 3);
  context.save();
  context.translate(x, y);
  context.rotate(angle);

  if (markType === 0) {
    context.scale(1.25 + Math.random() * 1.35, 0.62 + Math.random() * 0.52);
    const bruise = context.createRadialGradient(0, 0, radius * 0.08, 0, 0, radius);
    bruise.addColorStop(0, `rgba(75, 8, 18, ${0.18 + severity * 0.34})`);
    bruise.addColorStop(0.42, `rgba(116, 24, 30, ${0.12 + severity * 0.26})`);
    bruise.addColorStop(0.72, `rgba(61, 24, 62, ${0.08 + severity * 0.18})`);
    bruise.addColorStop(1, 'rgba(58, 20, 30, 0)');
    context.fillStyle = bruise;
    context.beginPath();
    context.arc(0, 0, radius, 0, Math.PI * 2);
    context.fill();
  } else if (markType === 1) {
    const length = radius * (2.2 + Math.random() * 2.4);
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.shadowColor = `rgba(52, 0, 0, ${0.3 + severity * 0.45})`;
    context.shadowBlur = 2 + severity * 6;
    context.strokeStyle = `rgba(91, 5, 10, ${0.35 + severity * 0.56})`;
    context.lineWidth = 1.2 + severity * 3.2;
    context.beginPath();
    context.moveTo(-length / 2, 0);
    for (let step = 1; step <= 5; step += 1) {
      context.lineTo(-length / 2 + (length * step) / 5, (Math.random() - 0.5) * radius * 0.42);
    }
    context.stroke();
    context.strokeStyle = `rgba(255, 133, 103, ${0.12 + severity * 0.24})`;
    context.lineWidth = Math.max(0.7, severity * 1.25);
    context.stroke();
  } else {
    const dots = 7 + Math.floor(Math.random() * 10);
    for (let dot = 0; dot < dots; dot += 1) {
      const dotRadius = radius * (0.07 + Math.random() * 0.16);
      context.fillStyle = `rgba(${70 + Math.floor(Math.random() * 55)}, ${12 + Math.floor(Math.random() * 25)}, ${12 + Math.floor(Math.random() * 28)}, ${0.12 + severity * (0.22 + Math.random() * 0.25)})`;
      context.beginPath();
      context.arc((Math.random() - 0.5) * radius * 2.3, (Math.random() - 0.5) * radius * 1.35, dotRadius, 0, Math.PI * 2);
      context.fill();
    }
  }

  context.restore();
  texture.needsUpdate = true;
}

function createCurvedFaceGeometry() {
  const geometry = new THREE.PlaneGeometry(0.71, 0.88, 32, 40);
  const position = geometry.attributes.position;
  for (let index = 0; index < position.count; index += 1) {
    const normalizedX = position.getX(index) / 0.355;
    const normalizedY = position.getY(index) / 0.44;
    const radius = Math.min(1, normalizedX * normalizedX + normalizedY * normalizedY);
    position.setZ(index, 0.085 * (1 - radius));
  }
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

function createCurvedTorsoGeometry() {
  const geometry = new THREE.PlaneGeometry(1.18, 0.96, 40, 32);
  const position = geometry.attributes.position;
  for (let index = 0; index < position.count; index += 1) {
    const originalX = position.getX(index);
    const normalizedY = position.getY(index) / 0.48;
    const widthScale = 0.88 + smoothstep(-0.8, 0.75, normalizedY) * 0.12;
    const x = originalX * widthScale;
    const normalizedX = x / 0.59;
    position.setX(index, x);
    position.setZ(index, 0.115 * (1 - normalizedX * normalizedX));
  }
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

function createMannequin(): { rig: MannequinRig; materials: ThreeTypes.MeshStandardMaterial[] } {
  const materials: ThreeTypes.MeshStandardMaterial[] = [];
  const material = (color = 0xe8e6e1) => {
    const next = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.78,
      metalness: 0.015,
    });
    materials.push(next);
    return next;
  };
  const mesh = (
    geometry: ThreeTypes.BufferGeometry,
    parent: ThreeTypes.Object3D,
    position: [number, number, number],
    scale: [number, number, number] = [1, 1, 1],
    color?: number,
  ) => {
    const part = new THREE.Mesh(geometry, material(color));
    part.position.set(...position);
    part.scale.set(...scale);
    part.castShadow = true;
    part.receiveShadow = true;
    parent.add(part);
    return part;
  };

  const root = new THREE.Group();
  root.position.set(0, 0, 0);

  const torso = new THREE.Group();
  torso.position.set(0, -0.35, 0);
  root.add(torso);
  mesh(new THREE.CapsuleGeometry(0.44, 0.28, 12, 32), torso, [0, -0.1, 0], [1.18, 1.12, 0.7]);
  mesh(new THREE.SphereGeometry(0.48, 36, 24), torso, [0, 0.34, 0], [1.48, 0.56, 0.7]);
  mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.22, 24), root, [0, 0.25, 0]);

  const createArm = (side: -1 | 1) => {
    const arm = new THREE.Group();
    arm.position.set(side * 0.65, 0.25, -0.01);
    arm.rotation.z = side * -0.07;
    torso.add(arm);
    mesh(new THREE.CapsuleGeometry(0.12, 0.36, 10, 24), arm, [0, -0.28, 0], [1, 1, 0.9]);
    const forearm = new THREE.Group();
    forearm.position.set(side * 0.025, -0.54, 0);
    forearm.rotation.z = side * 0.055;
    arm.add(forearm);
    mesh(new THREE.CapsuleGeometry(0.125, 0.34, 10, 24), forearm, [0, -0.2, 0], [1, 1, 0.9]);
    mesh(new THREE.SphereGeometry(0.13, 24, 18), forearm, [0, -0.47, 0], [0.88, 1.08, 0.84]);
    return arm;
  };

  const leftArm = createArm(-1);
  const rightArm = createArm(1);

  const head = new THREE.Group();
  head.position.set(0, 0.72, 0);
  root.add(head);
  mesh(new THREE.SphereGeometry(0.42, 36, 28), head, [0, 0, 0], [1, 1.13, 0.94]);

  return {
    rig: {
      root,
      torso,
      head,
      leftArm,
      rightArm,
    },
    materials,
  };
}

export function FightLab() {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);
  const menuAudioRef = useRef<HTMLAudioElement>(null);
  const gameAudioRef = useRef<HTMLAudioElement>(null);
  const readyAudioRef = useRef<HTMLAudioElement>(null);
  const punchAudioRefs = useRef<Array<HTMLAudioElement | null>>([]);
  const lastPunchSoundRef = useRef(-1);
  const targetRef = useRef<ThreeTypes.Group | null>(null);
  const rigRef = useRef<MannequinRig | null>(null);
  const faceRef = useRef<ThreeTypes.Mesh<ThreeTypes.PlaneGeometry, ThreeTypes.MeshStandardMaterial> | null>(null);
  const pendingFaceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mannequinMaterialsRef = useRef<ThreeTypes.MeshStandardMaterial[]>([]);
  const damageSystemRef = useRef<DamageSystem | null>(null);
  const damageHitCountRef = useRef(0);
  const hitRef = useRef<{ at: number; direction: HitDirection } | null>(null);
  const hitMotionRef = useRef<HitMotion>(createHitMotion());
  const poseRef = useRef<PoseLandmarkerInstance | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraFrameRef = useRef(0);
  const comboTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shakeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const goalTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const photoUrlRef = useRef<string | null>(null);
  const punchStateRef = useRef<PunchTracker>(createPunchTracker());
  const faceDetectorRef = useRef<FaceDetectorInstance | null>(null);
  const imageSegmenterRef = useRef<ImageSegmenterInstance | null>(null);
  const photoVisionPromiseRef = useRef<Promise<void> | null>(null);
  const gamePhaseRef = useRef<GamePhase>('menu');
  const comboRef = useRef(0);
  const roundPunchesRef = useRef(0);

  const [photoName, setPhotoName] = useState('');
  const [photoUrl, setPhotoUrl] = useState('');
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraMessage, setCameraMessage] = useState('摄像头未开启');
  const [poseVisible, setPoseVisible] = useState(false);
  const poseVisibleRef = useRef(false);
  const [combo, setCombo] = useState(0);
  const [lastHit, setLastHit] = useState<HitDirection>('right');
  const [impactNonce, setImpactNonce] = useState(0);
  const [shaking, setShaking] = useState(false);
  const [photoProcessing, setPhotoProcessing] = useState(false);
  const [arenaReady, setArenaReady] = useState(false);
  const [threeReady, setThreeReady] = useState(false);
  const [gamePhase, setGamePhase] = useState<GamePhase>('menu');
  const [gameMode, setGameMode] = useState<GameMode>('standard');
  const [introStep, setIntroStep] = useState<IntroStep>('waiting');
  const [selectedBackground, setSelectedBackground] = useState<string>(arenaBackgrounds[0].src);
  const [timeLeft, setTimeLeft] = useState(roundSeconds);
  const [roundPunches, setRoundPunches] = useState(0);
  const [bestCombo, setBestCombo] = useState(0);
  const [goalCelebrating, setGoalCelebrating] = useState(false);
  const [muted, setMuted] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [backgroundOpen, setBackgroundOpen] = useState(false);
  const [soundReady, setSoundReady] = useState(false);
  const shownCombo = Math.max(combo, 1);
  const impactWord = shownCombo >= 10
    ? '制霸!'
    : impactWords[(impactNonce + shownCombo - 1) % impactWords.length];
  const impactCaption = shownCombo >= 10
    ? '十连暴击 · 火力全开'
    : shownCombo >= 8
      ? '连续压制 · 不要停'
      : shownCombo >= 6
        ? '节奏拉满 · 越打越爽'
        : shownCombo >= 4
          ? '漂亮连击 · 继续输出'
          : shownCombo >= 2
            ? '连击开始 · 再来一拳'
            : '命中目标';

  useEffect(() => {
    const menuAudio = menuAudioRef.current;
    const gameAudio = gameAudioRef.current;
    if (!menuAudio || !gameAudio) return;
    menuAudio.volume = 0.48;
    gameAudio.volume = 0.42;
    if (readyAudioRef.current) readyAudioRef.current.volume = 0.95;
    void menuAudio.play()
      .then(() => setSoundReady(true))
      .catch(() => setSoundReady(false));
    return () => {
      menuAudio.pause();
      gameAudio.pause();
    };
  }, []);

  useEffect(() => {
    if (menuAudioRef.current) menuAudioRef.current.muted = muted;
    if (gameAudioRef.current) gameAudioRef.current.muted = muted;
    if (readyAudioRef.current) readyAudioRef.current.muted = muted;
  }, [muted]);

  useEffect(() => {
    if (gamePhase !== 'playing' || cameraState !== 'ready' || timeLeft <= 0) return;
    const timer = window.setInterval(() => {
      if (timeLeft <= 1) {
        setTimeLeft(0);
        gamePhaseRef.current = 'ended';
        setGamePhase('ended');
        setCombo(0);
        comboRef.current = 0;
      } else {
        setTimeLeft(timeLeft - 1);
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [cameraState, gamePhase, timeLeft]);

  useEffect(() => {
    if (gamePhase !== 'intro' || cameraState !== 'ready') return;
    const readyAudio = readyAudioRef.current;
    const gameAudio = gameAudioRef.current;
    if (gameAudio) gameAudio.volume = 0.18;
    if (readyAudio) {
      readyAudio.currentTime = 0;
      void readyAudio.play().catch(() => undefined);
    }
    const readyTimer = window.setTimeout(() => setIntroStep('ready'), 0);
    const fightTimer = window.setTimeout(() => setIntroStep('fight'), readyFightAtMs);
    const playTimer = window.setTimeout(() => {
      if (gameAudio) gameAudio.volume = 0.42;
      gamePhaseRef.current = 'playing';
      setGamePhase('playing');
    }, readyClipDurationMs);
    return () => {
      window.clearTimeout(readyTimer);
      window.clearTimeout(fightTimer);
      window.clearTimeout(playTimer);
      readyAudio?.pause();
      if (gameAudio) gameAudio.volume = 0.42;
    };
  }, [cameraState, gamePhase]);

  const enableMenuSound = useCallback(() => {
    const menuAudio = menuAudioRef.current;
    if (!menuAudio) return;
    setMuted(false);
    void menuAudio.play().then(() => setSoundReady(true)).catch(() => setSoundReady(false));
  }, []);

  const setSoundEnabled = useCallback((enabled: boolean) => {
    setMuted(!enabled);
    if (!enabled) return;
    const activeAudio = gamePhaseRef.current === 'menu' ? menuAudioRef.current : gameAudioRef.current;
    void activeAudio?.play().then(() => setSoundReady(true)).catch(() => setSoundReady(false));
  }, []);

  const startRound = useCallback((mode: GameMode) => {
    const menuAudio = menuAudioRef.current;
    const gameAudio = gameAudioRef.current;
    setArenaReady(true);
    menuAudio?.pause();
    readyAudioRef.current?.load();
    punchAudioRefs.current.forEach((audio) => audio?.load());
    if (gameAudio) {
      gameAudio.load();
      gameAudio.currentTime = 0;
      if (!muted) void gameAudio.play().then(() => setSoundReady(true)).catch(() => undefined);
    }
    if (faceRef.current) {
      faceRef.current.material.opacity = mode === 'practice' ? 0 : pendingFaceCanvasRef.current ? 1 : 0;
      faceRef.current.material.needsUpdate = true;
    }
    if (comboTimerRef.current) clearTimeout(comboTimerRef.current);
    if (goalTimerRef.current) clearTimeout(goalTimerRef.current);
    comboRef.current = 0;
    roundPunchesRef.current = 0;
    setCombo(0);
    setRoundPunches(0);
    setBestCombo(0);
    setGoalCelebrating(false);
    setTimeLeft(roundSeconds);
    setGameMode(mode);
    setIntroStep('waiting');
    gamePhaseRef.current = 'intro';
    setGamePhase('intro');
  }, [muted]);

  const applyFaceTexture = useCallback((canvas: HTMLCanvasElement) => {
    pendingFaceCanvasRef.current = canvas;
    const face = faceRef.current;
    if (!face) return;
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    face.material.map?.dispose();
    face.material.map = texture;
    face.material.opacity = 1;
    face.material.needsUpdate = true;
  }, []);

  const clearDamage = useCallback(() => {
    damageHitCountRef.current = 0;
    const damage = damageSystemRef.current;
    if (!damage) return;
    [damage.face, damage.body].forEach((surface) => {
      surface.context.clearRect(0, 0, surface.canvas.width, surface.canvas.height);
      surface.texture.needsUpdate = true;
    });
  }, []);

  const ensurePhotoVision = useCallback(async () => {
    if (faceDetectorRef.current && imageSegmenterRef.current) return;
    if (!photoVisionPromiseRef.current) {
      photoVisionPromiseRef.current = (async () => {
        const { FaceDetector, FilesetResolver, ImageSegmenter } = await import('@mediapipe/tasks-vision');
        const fileset = await FilesetResolver.forVisionTasks('/wasm');
        faceDetectorRef.current = await FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/models/blaze-face-short-range.tflite', delegate: 'GPU' },
          runningMode: 'IMAGE',
          minDetectionConfidence: 0.35,
        });
        imageSegmenterRef.current = await ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/models/selfie-segmenter.tflite', delegate: 'GPU' },
          runningMode: 'IMAGE',
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        });
      })().catch(async () => {
        faceDetectorRef.current?.close();
        imageSegmenterRef.current?.close();
        faceDetectorRef.current = null;
        imageSegmenterRef.current = null;
        const { FaceDetector, FilesetResolver, ImageSegmenter } = await import('@mediapipe/tasks-vision');
        const fileset = await FilesetResolver.forVisionTasks('/wasm');
        faceDetectorRef.current = await FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/models/blaze-face-short-range.tflite', delegate: 'CPU' },
          runningMode: 'IMAGE',
          minDetectionConfidence: 0.35,
        });
        imageSegmenterRef.current = await ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/models/selfie-segmenter.tflite', delegate: 'CPU' },
          runningMode: 'IMAGE',
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        });
      });
    }
    await photoVisionPromiseRef.current;
  }, []);

  const playImpactSound = useCallback(() => {
    if (muted) return;
    const available = punchAudioRefs.current.filter((audio): audio is HTMLAudioElement => Boolean(audio));
    if (!available.length) return;
    let nextIndex = Math.floor(Math.random() * available.length);
    if (available.length > 1 && nextIndex === lastPunchSoundRef.current) {
      nextIndex = (nextIndex + 1 + Math.floor(Math.random() * (available.length - 1))) % available.length;
    }
    lastPunchSoundRef.current = nextIndex;
    const sound = available[nextIndex];
    sound.currentTime = 0;
    sound.volume = 0.94;
    void sound.play().catch(() => undefined);
  }, [muted]);

  const addDamageMark = useCallback((hitIndex: number) => {
    const damage = damageSystemRef.current;
    if (!damage) return;
    const surface = hitIndex % 5 === 0 || Math.random() < 0.43 ? damage.face : damage.body;
    paintDamage(surface, hitIndex);
  }, []);

  const triggerHit = useCallback(
    (direction: HitDirection) => {
      if (gamePhaseRef.current !== 'playing') return;
      hitRef.current = { at: performance.now(), direction };
      const sign = direction === 'left' ? 1 : -1;
      const motion = hitMotionRef.current;
      motion.headRoll.velocity += sign * 8.6;
      motion.headYaw.velocity += sign * 10.4;
      motion.headShift.velocity += sign * 3.2;
      motion.torsoRoll.velocity += sign * 3.8;
      motion.torsoYaw.velocity += sign * 4.8;
      motion.torsoShift.velocity += sign * 1.35;
      motion.depth.velocity -= 2.2;
      setLastHit(direction);
      setImpactNonce((value) => value + 1);
      setShaking(true);
      if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
      shakeTimerRef.current = setTimeout(() => setShaking(false), 300);
      const nextCombo = comboRef.current + 1;
      comboRef.current = nextCombo;
      setCombo(nextCombo);
      setBestCombo((value) => Math.max(value, nextCombo));
      const nextPunches = roundPunchesRef.current + 1;
      roundPunchesRef.current = nextPunches;
      setRoundPunches(nextPunches);
      damageHitCountRef.current += 1;
      addDamageMark(damageHitCountRef.current);
      if (nextPunches === targetPunches) {
        setGoalCelebrating(true);
        if (goalTimerRef.current) clearTimeout(goalTimerRef.current);
        goalTimerRef.current = setTimeout(() => setGoalCelebrating(false), 2100);
      }
      if (comboTimerRef.current) clearTimeout(comboTimerRef.current);
      comboTimerRef.current = setTimeout(() => {
        comboRef.current = 0;
        setCombo(0);
      }, 1500);
      playImpactSound();
      navigator.vibrate?.(28);
    },
    [addDamageMark, playImpactSound],
  );

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(cameraFrameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    poseRef.current?.close();
    poseRef.current = null;
    punchStateRef.current = createPunchTracker();
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState('idle');
    setCameraMessage('摄像头未开启');
    setPoseVisible(false);
    poseVisibleRef.current = false;
  }, []);

  const returnToMenu = useCallback(() => {
    stopCamera();
    if (comboTimerRef.current) clearTimeout(comboTimerRef.current);
    comboRef.current = 0;
    setCombo(0);
    setIntroStep('waiting');
    setGoalCelebrating(false);
    gamePhaseRef.current = 'menu';
    setGamePhase('menu');
    gameAudioRef.current?.pause();
    readyAudioRef.current?.pause();
    if (!muted) {
      void menuAudioRef.current?.play().then(() => setSoundReady(true)).catch(() => undefined);
    }
  }, [muted, stopCamera]);

  const changePhotoAfterRound = useCallback(() => {
    returnToMenu();
    photoInputRef.current?.click();
  }, [returnToMenu]);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setCameraMessage('当前浏览器不支持摄像头');
      return;
    }
    setCameraState('loading');
    setCameraMessage('正在准备识别…');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      const activeVideo = video;
      activeVideo.srcObject = stream;
      await activeVideo.play();

      const { FilesetResolver, PoseLandmarker } = await import('@mediapipe/tasks-vision');
      const fileset = await FilesetResolver.forVisionTasks('/wasm');
      const options = {
        runningMode: 'VIDEO' as const,
        numPoses: 1,
        minPoseDetectionConfidence: 0.42,
        minPosePresenceConfidence: 0.42,
        minTrackingConfidence: 0.42,
      };
      try {
        poseRef.current = await PoseLandmarker.createFromOptions(fileset, {
          ...options,
          baseOptions: { modelAssetPath: '/models/pose-landmarker-lite.task', delegate: 'GPU' },
        });
      } catch {
        poseRef.current = await PoseLandmarker.createFromOptions(fileset, {
          ...options,
          baseOptions: { modelAssetPath: '/models/pose-landmarker-lite.task', delegate: 'CPU' },
        });
      }

      punchStateRef.current = createPunchTracker();
      setCameraState('ready');
      setCameraMessage('收拳后快速向前出拳');
      let lastDetection = 0;
      function detect() {
        // oxlint-disable-next-line react/react-compiler -- the animation callback intentionally schedules itself.
        cameraFrameRef.current = requestAnimationFrame(detect);
        const landmarker = poseRef.current;
        if (!landmarker || activeVideo.readyState < 2) return;
        const now = performance.now();
        if (now - lastDetection < 45) return;
        lastDetection = now;
        const result = landmarker.detectForVideo(activeVideo, now);
        const landmarks = result.landmarks?.[0] as Landmark[] | undefined;
        const isVisible = Boolean(landmarks?.length);
        if (isVisible !== poseVisibleRef.current) {
          poseVisibleRef.current = isVisible;
          setPoseVisible(isVisible);
        }
        if (!landmarks) {
          punchStateRef.current = createPunchTracker();
          return;
        }
        const punch = findPunch(landmarks, now, punchStateRef.current);
        if (punch) triggerHit(punch);
      }
      detect();
    } catch (error) {
      stopCamera();
      setCameraState('error');
      setCameraMessage(error instanceof DOMException && error.name === 'NotAllowedError' ? '请允许摄像头权限' : '摄像头启动失败');
    }
  }, [stopCamera, triggerHit]);

  const handlePhoto = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const input = event.target;
      const file = input.files?.[0];
      input.value = '';
      if (!file || !file.type.startsWith('image/')) return;
      if (file.size > 10 * 1024 * 1024) {
        setPhotoName('图片需小于 10MB');
        return;
      }
      setPhotoProcessing(true);
      setPhotoName('正在抠出完整头部…');
      let image: ImageBitmap | null = null;
      try {
        image = await createImageBitmap(file);
        const normalized = createNormalizedPhoto(image);
        let faceBox: BoundingBox | undefined;
        let personMask: { data: Float32Array; width: number; height: number } | undefined;
        try {
          await ensurePhotoVision();
          const detections = faceDetectorRef.current?.detect(normalized).detections ?? [];
          faceBox = detections
            .map((detection) => detection.boundingBox)
            .filter((box): box is BoundingBox => Boolean(box))
            .sort((a, b) => b.width * b.height - a.width * a.height)[0];
          const segmentResult = imageSegmenterRef.current?.segment(normalized);
          const confidenceMask = segmentResult?.confidenceMasks?.[0];
          if (confidenceMask) {
            personMask = {
              data: confidenceMask.getAsFloat32Array(),
              width: confidenceMask.width,
              height: confidenceMask.height,
            };
          }
          segmentResult?.confidenceMasks?.forEach((mask) => mask.close());
        } catch {
          photoVisionPromiseRef.current = null;
        }
        clearDamage();
        applyFaceTexture(createHeadTexture(normalized, faceBox, personMask));
        if (photoUrlRef.current) URL.revokeObjectURL(photoUrlRef.current);
        const nextUrl = URL.createObjectURL(file);
        photoUrlRef.current = nextUrl;
        setPhotoUrl(nextUrl);
        setPhotoName(personMask ? '已删除背景 · 保留完整头部' : '已按头部轮廓去除背景');
      } catch {
        setPhotoName('无法读取这张图片，请换 JPG / PNG');
      } finally {
        image?.close();
        setPhotoProcessing(false);
      }
    },
    [applyFaceTexture, clearDamage, ensurePhotoVision],
  );

  useEffect(() => {
    if (!arenaReady) return;
    let active = true;
    void import('three').then((module) => {
      THREE = module;
      if (active) setThreeReady(true);
    });
    return () => {
      active = false;
    };
  }, [arenaReady]);

  useEffect(() => {
    if (!arenaReady || !threeReady || !stageRef.current) return;
    const host = stageRef.current;
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x08080a, 0.065);
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    camera.position.set(0, 0.08, 4.35);
    camera.lookAt(0, 0.08, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.14;
    host.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xf8f8f8, 0x16161b, 2.7));
    const key = new THREE.DirectionalLight(0xff7555, 4.5);
    key.position.set(4, 5, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x7de8ff, 3.1);
    rim.position.set(-4, 2, 2);
    scene.add(rim);
    const { rig, materials } = createMannequin();
    mannequinMaterialsRef.current = materials;
    const face = new THREE.Mesh(
      createCurvedFaceGeometry(),
      new THREE.MeshStandardMaterial({
        transparent: true,
        opacity: 0,
        alphaTest: 0.025,
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
        roughness: 0.76,
        metalness: 0,
      }),
    );
    face.position.set(0, 0, 0.4);
    face.renderOrder = 999;
    rig.head.add(face);
    const bodyMarkTexture = new THREE.CanvasTexture(createBodyMarkTexture());
    bodyMarkTexture.colorSpace = THREE.SRGBColorSpace;
    const bodyMark = new THREE.Mesh(
      createCurvedTorsoGeometry(),
      new THREE.MeshStandardMaterial({
        map: bodyMarkTexture,
        transparent: true,
        opacity: 1,
        alphaTest: 0.02,
        depthTest: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
        side: THREE.DoubleSide,
        roughness: 0.8,
        metalness: 0,
      }),
    );
    bodyMark.position.set(0, 0.04, 0.228);
    bodyMark.renderOrder = 998;
    rig.torso.add(bodyMark);

    const faceDamageSurface = createDamageSurface(420, 500);
    const bodyDamageSurface = createDamageSurface(640, 520);
    const faceDamageMaterial = new THREE.MeshBasicMaterial({
      map: faceDamageSurface.texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      alphaTest: 0.005,
      side: THREE.DoubleSide,
    });
    const bodyDamageMaterial = new THREE.MeshBasicMaterial({
      map: bodyDamageSurface.texture,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      alphaTest: 0.005,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      side: THREE.DoubleSide,
    });
    const faceDamage = new THREE.Mesh(createCurvedFaceGeometry(), faceDamageMaterial);
    faceDamage.position.set(0, 0, 0.414);
    faceDamage.renderOrder = 1002;
    rig.head.add(faceDamage);
    const bodyDamage = new THREE.Mesh(createCurvedTorsoGeometry(), bodyDamageMaterial);
    bodyDamage.position.set(0, 0.04, 0.235);
    bodyDamage.renderOrder = 1001;
    rig.torso.add(bodyDamage);
    damageSystemRef.current = { face: faceDamageSurface, body: bodyDamageSurface };

    scene.add(rig.root);
    targetRef.current = rig.root;
    rigRef.current = rig;
    faceRef.current = face;
    if (pendingFaceCanvasRef.current) applyFaceTexture(pendingFaceCanvasRef.current);

    let frame = 0;
    let lastFrameAt = performance.now();
    const resize = () => {
      const { clientWidth, clientHeight } = host;
      renderer.setSize(clientWidth, clientHeight);
      camera.aspect = clientWidth / Math.max(clientHeight, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      const delta = Math.min((now - lastFrameAt) / 1000, 0.033);
      lastFrameAt = now;
      const target = targetRef.current;
      const rig = rigRef.current;
      if (target && rig) {
        const motion = hitMotionRef.current;
        stepSpring(motion.headRoll, 96, 14.5, delta);
        stepSpring(motion.headYaw, 112, 15.5, delta);
        stepSpring(motion.headShift, 118, 18, delta);
        stepSpring(motion.torsoRoll, 78, 15.5, delta);
        stepSpring(motion.torsoYaw, 88, 16, delta);
        stepSpring(motion.torsoShift, 92, 17, delta);
        stepSpring(motion.depth, 105, 19, delta);

        rig.head.rotation.z = motion.headRoll.position;
        rig.head.rotation.y = motion.headYaw.position;
        rig.head.position.x = motion.headShift.position;
        rig.head.position.y = 0.72 - Math.abs(motion.headRoll.position) * 0.035;
        rig.torso.rotation.z = motion.torsoRoll.position;
        rig.torso.rotation.y = motion.torsoYaw.position;
        rig.torso.position.x = motion.torsoShift.position;
        rig.torso.position.y = -0.35;
        target.position.z = motion.depth.position;

        const hit = hitRef.current;
        const flash = hit ? clamp01(1 - (now - hit.at) / 180) : 0;
        mannequinMaterialsRef.current.forEach((material) => {
          material.emissive.setHex(0xff3b24);
          material.emissiveIntensity = flash * 1.7;
        });
        if (hit && now - hit.at > 190) hitRef.current = null;
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.dispose();
      targetRef.current?.traverse((child) => {
        if (child instanceof THREE.Mesh && child !== faceRef.current) child.geometry.dispose();
      });
      faceRef.current?.geometry.dispose();
      faceRef.current?.material.map?.dispose();
      faceRef.current?.material.dispose();
      bodyMark.material.map?.dispose();
      bodyMark.material.dispose();
      faceDamageSurface.texture.dispose();
      bodyDamageSurface.texture.dispose();
      faceDamageMaterial.dispose();
      bodyDamageMaterial.dispose();
      mannequinMaterialsRef.current.forEach((material) => material.dispose());
      mannequinMaterialsRef.current = [];
      renderer.domElement.remove();
      targetRef.current = null;
      rigRef.current = null;
      hitMotionRef.current = createHitMotion();
      faceRef.current = null;
      damageSystemRef.current = null;
    };
  }, [applyFaceTexture, arenaReady, threeReady]);

  useEffect(() => {
    const readyAudio = readyAudioRef.current;
    const punchAudios = [...punchAudioRefs.current];
    return () => {
      stopCamera();
      if (comboTimerRef.current) clearTimeout(comboTimerRef.current);
      if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
      if (goalTimerRef.current) clearTimeout(goalTimerRef.current);
      if (photoUrlRef.current) URL.revokeObjectURL(photoUrlRef.current);
      readyAudio?.pause();
      punchAudios.forEach((audio) => audio?.pause());
      faceDetectorRef.current?.close();
      imageSegmenterRef.current?.close();
      faceDetectorRef.current = null;
      imageSegmenterRef.current = null;
      photoVisionPromiseRef.current = null;
    };
  }, [stopCamera]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-background p-2 text-foreground sm:p-3">
      <audio ref={menuAudioRef} src="/audio/menu.mp3" loop preload="none" autoPlay>
        <track kind="captions" src="/audio/no-dialogue.vtt" srcLang="zh" label="纯音乐，无对白" default />
      </audio>
      <audio ref={gameAudioRef} src="/audio/game.mp3" loop preload="none">
        <track kind="captions" src="/audio/no-dialogue.vtt" srcLang="zh" label="纯音乐，无对白" default />
      </audio>
      <audio ref={readyAudioRef} src="/audio/ready.mp3" preload="none">
        <track kind="captions" src="/audio/ready.vtt" srcLang="en" label="READY FIGHT 开场语音" default />
      </audio>
      {punchSoundSources.map((source, index) => (
        <audio
          key={source}
          ref={(node) => { punchAudioRefs.current[index] = node; }}
          src={source}
          preload="none"
        >
          <track kind="captions" src="/audio/no-dialogue.vtt" srcLang="zh" label={`拳击音效 ${index + 1}`} default />
        </audio>
      ))}
      <input ref={photoInputRef} className="sr-only" type="file" accept="image/*" onChange={handlePhoto} disabled={photoProcessing} />

      <section className={`minimal-arena relative min-h-[calc(100dvh-1rem)] overflow-hidden rounded-[26px] sm:min-h-[calc(100dvh-1.5rem)] ${shaking ? 'arena-shake' : ''}`}>
        {arenaReady ? <div className="arena-background" style={{ backgroundImage: `url(${selectedBackground})` }} aria-hidden="true" /> : null}
        <div className="arena-vignette" aria-hidden="true" />
        <div ref={stageRef} className="absolute inset-0 z-[2]" aria-label="2.5D 训练目标" />

        {gamePhase !== 'menu' ? (
          <button type="button" className="game-back" onClick={returnToMenu}>
            <ArrowLeft aria-hidden="true" /> 返回
          </button>
        ) : null}

        {gamePhase !== 'menu' && gameMode === 'standard' ? (
          <button type="button" onClick={() => photoInputRef.current?.click()} disabled={photoProcessing} className={`game-photo-upload ${photoProcessing ? 'is-processing' : ''}`}>
            <span className="game-photo-upload__preview">
              {photoUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={photoUrl} alt="已选择的本地照片" className="size-full object-cover" />
              ) : (
                <ImagePlus className="size-5 text-white/45" />
              )}
            </span>
            <span className="game-photo-upload__copy">
              <span className="game-photo-upload__title">
                <Upload aria-hidden="true" /> {photoProcessing ? '正在识别' : photoUrl ? '更换照片' : '上传照片'}
              </span>
              <span className="game-photo-upload__meta">{photoName || '上传正脸照片 · 10MB 内'}</span>
            </span>
          </button>
        ) : null}

        {gamePhase === 'intro' || gamePhase === 'playing' ? (
          <div className="round-timer" aria-live="polite">
            <span>{gameMode === 'practice' ? 'PRACTICE' : 'TIME LIMIT'}</span>
            <strong>{String(timeLeft).padStart(2, '0')}</strong>
            <small>{cameraState === 'ready' ? 'SECONDS' : '开启摄像头后开始'}</small>
          </div>
        ) : null}

        {gamePhase === 'intro' || gamePhase === 'playing' ? (
          <div
            className={`punch-gauge ${roundPunches > 0 ? 'is-charged' : ''} ${roundPunches > targetPunches ? 'is-overdrive' : ''}`}
            style={{
              '--rumble-speed': `${Math.max(90, 430 - roundPunches * 5)}ms`,
              '--punch-progress': `${Math.min(100, (roundPunches / targetPunches) * 100)}%`,
            } as CSSProperties}
            aria-label={`拳击数量 ${roundPunches} / ${targetPunches}`}
          >
            {roundPunches > targetPunches ? (
              <div className="punch-gauge__flames" aria-hidden="true"><b /><b /><b /><b /><b /></div>
            ) : null}
            <div className="punch-gauge__art" aria-hidden="true">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="punch-gauge__art-empty" src="/ui/punch-empty.jpg" alt="" />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="punch-gauge__art-fill" src="/ui/punch-fill.jpg" alt="" />
            </div>
            <div className="punch-gauge__header">PUNCH</div>
            <strong>{String(roundPunches).padStart(2, '0')}</strong>
            <small>/ {targetPunches}</small>
          </div>
        ) : null}

        {gamePhase === 'intro' && cameraState === 'ready' && introStep !== 'waiting' ? (
          <div key={introStep} className={`ready-go ready-go--${introStep}`} aria-live="assertive">
            {introStep === 'ready' ? 'READY' : 'FIGHT!'}
          </div>
        ) : null}

        {goalCelebrating ? (
          <div className="goal-success" aria-live="assertive">
            <strong>成功!</strong>
            <span>50 PUNCHES CLEARED</span>
          </div>
        ) : null}

        {impactNonce > 0 ? (
          <div key={impactNonce} className="pointer-events-none absolute inset-0 z-10 overflow-hidden" aria-hidden="true">
            <div className="impact-flash" />
            <div className={`impact-copy ${lastHit === 'left' ? '-rotate-6' : 'rotate-6'}`}>
              <div className="impact-mainline">
                <span className="impact-word">{impactWord}</span>
                <span className="impact-multiplier">×{shownCombo}</span>
              </div>
              <div className="impact-subtitle">{impactCaption}</div>
            </div>
            <div className="impact-burst">
              {Array.from({ length: 18 }).map((_, index) => {
                const angle = (Math.PI * 2 * index) / 18 + impactNonce * 0.23;
                const distance = 90 + (index % 5) * 23;
                const style = {
                  '--dx': `${Math.cos(angle) * distance}px`,
                  '--dy': `${Math.sin(angle) * distance}px`,
                  '--rot': `${index * 47}deg`,
                  '--particle': particleColors[index % particleColors.length],
                } as CSSProperties;
                return <span key={index} className="impact-particle" style={style} />;
              })}
            </div>
          </div>
        ) : null}

        {gamePhase !== 'menu' ? (
          <aside className="camera-panel">
            <div className="camera-panel__viewport">
              <video ref={videoRef} muted playsInline className={`size-full scale-x-[-1] object-cover transition-opacity ${cameraState === 'ready' ? 'opacity-100' : 'opacity-0'}`} />
              {cameraState !== 'ready' ? (
                <div className="camera-panel__placeholder"><Camera aria-hidden="true" /></div>
              ) : null}
              <div className="camera-panel__status">
                <span className={cameraState === 'ready' && poseVisible ? 'is-ready' : ''} />
                {cameraState === 'ready' ? (poseVisible ? '已锁定' : '寻找人物') : '摄像头'}
              </div>
            </div>
            {cameraState === 'ready' ? (
              <Button variant="outline" className="camera-panel__action" onClick={stopCamera}>
                <CameraOff /> 关闭摄像头
              </Button>
            ) : (
              <Button className="camera-panel__action" onClick={startCamera} disabled={cameraState === 'loading'}>
                <Camera /> {cameraState === 'loading' ? '准备中…' : '开启摄像头'}
              </Button>
            )}
            <p className={`camera-panel__message ${cameraState === 'error' ? 'is-error' : ''}`}>{cameraMessage}</p>
          </aside>
        ) : null}

        {gamePhase === 'ended' ? (
          <dialog open className="round-result" aria-labelledby="round-result-title">
            <div className={`round-result__panel ${roundPunches >= targetPunches ? 'is-win' : 'is-lose'}`}>
              <span className="round-result__eyebrow">本局结束</span>
              <h2 id="round-result-title">{roundPunches >= targetPunches ? '游戏胜利' : '挑战失败'}</h2>
              <p>{roundPunches >= targetPunches ? '目标达成，火力全开。' : `还差 ${targetPunches - roundPunches} 拳，再来一次。`}</p>
              <div className="round-result__stats">
                <div><span>最高连击</span><strong>×{bestCombo}</strong></div>
                <div><span>出拳次数</span><strong>{roundPunches}</strong></div>
                <div><span>游戏时长</span><strong>{roundSeconds - timeLeft}秒</strong></div>
              </div>
              <div className="round-result__actions">
                <button type="button" onClick={() => startRound(gameMode)}>再来一局</button>
                <button type="button" onClick={changePhotoAfterRound}>换一张照片</button>
                <button type="button" onClick={returnToMenu}>返回主界面</button>
              </div>
            </div>
          </dialog>
        ) : null}
      </section>

      <div className={`cyber-menu ${gamePhase !== 'menu' ? 'cyber-menu--hidden' : ''}`} aria-hidden={gamePhase !== 'menu'}>
        <div className="cyber-menu__glow" aria-hidden="true" />
        <div className="cyber-menu__grid" aria-hidden="true" />
        <div className="cyber-menu__scanlines" aria-hidden="true" />
        <button type="button" className="cyber-menu__settings" onClick={() => setSettingsOpen(true)} aria-label="打开设置">
          <Settings2 aria-hidden="true" />
        </button>
        <div className="cyber-menu__frame">
          <h1 className="cyber-menu__title">打小人 · PUNCH THE GRUDGE</h1>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="cyber-menu__logo" src="/branding/daxiaoren-logo.jpg" alt="打小人 · PUNCH THE GRUDGE" />
          <button type="button" className="cyber-menu__start" onClick={() => startRound('standard')}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/branding/arcade-button.jpg" alt="" aria-hidden="true" />
            <span>开始游戏</span>
          </button>
          <div className="cyber-menu__actions">
            <button type="button" className="cyber-menu__secondary" onClick={() => startRound('practice')}>
              <Dumbbell aria-hidden="true" />
              <span>练习模式</span>
              <ChevronRight aria-hidden="true" />
            </button>
            <button type="button" className="cyber-menu__secondary" onClick={() => setBackgroundOpen(true)}>
              <ImagePlus aria-hidden="true" />
              <span>选择背景</span>
              <ChevronRight aria-hidden="true" />
            </button>
          </div>
          {!soundReady ? (
            <button type="button" className="cyber-menu__sound" onClick={enableMenuSound}>
              <Volume2 aria-hidden="true" /> 开启声音
            </button>
          ) : null}
        </div>
      </div>

      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogContent className="cyber-dialog z-[140]" showCloseButton>
          <DialogHeader>
            <DialogTitle>设置</DialogTitle>
            <DialogDescription>控制主界面音乐和游戏音效。</DialogDescription>
          </DialogHeader>
          <div className="cyber-setting-row">
            <span>{muted ? <VolumeX aria-hidden="true" /> : <Volume2 aria-hidden="true" />} 游戏声音</span>
            <Switch checked={!muted} onCheckedChange={setSoundEnabled} aria-label="游戏声音" />
          </div>
          <button type="button" className="dialog-back" onClick={() => setSettingsOpen(false)}>
            <ArrowLeft aria-hidden="true" /> 返回主界面
          </button>
        </DialogContent>
      </Dialog>

      <Dialog open={backgroundOpen} onOpenChange={setBackgroundOpen}>
        <DialogContent className="cyber-dialog background-dialog z-[140]" showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>选择背景</DialogTitle>
            <DialogDescription>选择一个游戏场景，第一张“地下刑房”为默认背景。</DialogDescription>
          </DialogHeader>
          <div className="background-slots" aria-label="游戏背景">
            {arenaBackgrounds.map((background) => (
              <button
                key={background.id}
                type="button"
                className={selectedBackground === background.src ? 'is-selected' : ''}
                onClick={() => {
                  setSelectedBackground(background.src);
                  setBackgroundOpen(false);
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={background.src} alt={background.name} />
                <span>{background.name}</span>
              </button>
            ))}
          </div>
          <button type="button" className="dialog-back" onClick={() => setBackgroundOpen(false)}>
            <ArrowLeft aria-hidden="true" /> 返回主界面
          </button>
        </DialogContent>
      </Dialog>
    </main>
  );
}
