#!/usr/bin/env python3
"""Captura o movimento de VÁRIAS pessoas de um vídeo e gera um .vmd para cada uma.

Exemplo:
    python video2vmd.py danca.mp4                    # 2 pessoas -> danca_pessoa1.vmd, danca_pessoa2.vmd
    python video2vmd.py danca.mp4 --preview          # também gera danca_preview.mp4 para conferir quem é quem
    python video2vmd.py danca.mp4 --swap 450         # a partir do quadro 450 troca pessoa1 <-> pessoa2

Pessoa 1 é quem está mais à ESQUERDA no primeiro quadro em que todos aparecem.
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

from mpvmd import filters, retarget, tracker
from mpvmd.vmd import BoneFrame, write_vmd

VMD_FPS = 30
FEET = [27, 28, 29, 30, 31, 32]


def parse_swap(text):
    parts = text.split(":")
    if len(parts) == 1:
        return int(parts[0]), 0, 1
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]) - 1, int(parts[2]) - 1
    raise argparse.ArgumentTypeError("use QUADRO ou QUADRO:A:B (ex.: 450 ou 450:1:2)")


def build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="vídeo de entrada (.mp4, .mov, ...)")
    p.add_argument("-o", "--output", help="prefixo dos arquivos de saída (padrão: mesmo nome do vídeo)")
    p.add_argument("-n", "--persons", type=int, default=2, help="quantas pessoas capturar (padrão 2)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="heavy",
                   help="modelo MediaPipe: heavy é o mais preciso (padrão)")
    p.add_argument("--models-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    p.add_argument("--start", type=float, default=0.0, help="começar em N segundos")
    p.add_argument("--end", type=float, default=None, help="terminar em N segundos")
    p.add_argument("--det-conf", type=float, default=0.5, help="confiança mínima de detecção (0-1)")
    p.add_argument("--reextract", action="store_true",
                   help="ignora o cache .landmarks.pkl e roda a detecção de novo")

    g = p.add_argument_group("identidade das pessoas")
    g.add_argument("--swap", type=parse_swap, action="append", default=[],
                   help="troca identidades a partir de um quadro do preview: QUADRO (troca 1<->2) ou QUADRO:A:B. "
                        "Pode repetir.")
    g.add_argument("--color-weight", type=float, default=1.0,
                   help="peso da cor da roupa ao diferenciar as pessoas (0 desliga)")
    g.add_argument("--preview", action="store_true", help="gera um vídeo com o esqueleto e o número de cada pessoa")

    g = p.add_argument_group("movimento")
    g.add_argument("--arm-angle", type=float, default=35.0,
                   help="ângulo (graus) dos braços do modelo em pose de descanso (A-pose). "
                        "Se os braços ficarem altos/baixos demais, ajuste (30-45)")
    g.add_argument("--smooth", type=float, default=1.0,
                   help="suavização: maior = mais suave e com mais atraso (0 desliga)")
    g.add_argument("--scale", type=float, default=12.5, help="unidades MMD por metro (padrão 12.5)")
    g.add_argument("--fov", type=float, default=60.0, help="campo de visão vertical estimado da câmera (graus)")
    g.add_argument("--depth", type=float, default=1.0,
                   help="quanto do deslocamento para frente/trás usar (0 = ninguém se move em profundidade)")
    g.add_argument("--no-translation", action="store_true", help="não move a センター (só rotações)")
    g.add_argument("--origin", choices=["shared", "each"], default="shared",
                   help="shared: posições relativas a um ponto comum (mantém a distância entre as pessoas); "
                        "each: cada pessoa começa no próprio ponto zero")
    g.add_argument("--no-wrists", action="store_true", help="não anima os pulsos (os pontos das mãos são ruidosos)")
    g.add_argument("--no-ankles", action="store_true", help="não anima os tornozelos")
    g.add_argument("--keep-ik", action="store_true",
                   help="não desliga o IK das pernas no .vmd (por padrão é desligado para usar as rotações FK)")
    g.add_argument("--max-gap", type=int, default=15,
                   help="interpola buracos de até N quadros onde a pessoa sumiu (padrão 15)")
    return p


def load_or_extract(args, cache_path):
    if os.path.exists(cache_path) and not args.reextract:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("persons") == args.persons and data.get("start") == args.start and data.get("end") == args.end:
            print(f"Usando detecções em cache: {cache_path} (use --reextract para refazer)")
            return data["frames"], data["times"], data["info"]
    from mpvmd import extract

    print(f"Detectando até {args.persons + 1} pessoas por quadro com MediaPipe ({args.model})...")
    # Ask for one extra pose so a false positive doesn't push a real dancer out.
    frames, times, info = extract.extract(args.video, args.models_dir, args.model, args.persons + 1,
                                          det_conf=args.det_conf, start=args.start, end=args.end)
    with open(cache_path, "wb") as f:
        pickle.dump({"frames": frames, "times": times, "info": info, "persons": args.persons,
                     "start": args.start, "end": args.end}, f)
    return frames, times, info


def camera_position(det, width, height, fov_deg):
    """Estimate the hip centre in camera space (metres, MMD axes).

    MediaPipe's world landmarks are centred on the hips, so they say nothing
    about where the person stands. We recover it from the image: comparing the
    metric (x, y) length of several body segments with their length in pixels
    gives the distance to the camera, as in a pinhole camera.
    """
    f = (height / 2) / math.tan(math.radians(fov_deg) / 2)
    segs = [(11, 12), (23, 24), (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
    w, px, vis = det["world"], det["image_px"], det["vis"]
    metric = pixels = 0.0
    for a, b in segs:
        wt = min(vis[a], vis[b])
        metric += wt * np.linalg.norm(w[a, :2] - w[b, :2])
        pixels += wt * np.linalg.norm(px[a, :2] - px[b, :2])
    if pixels < 1e-6:
        return None
    z = f * metric / pixels
    u, v = (px[23, :2] + px[24, :2]) / 2
    return np.array([(u - width / 2) * z / f, -(v - height / 2) * z / f, z])


def build_person_arrays(assigned, k, info, fov):
    width, height = info[1], info[2]
    T = len(assigned)
    world = np.zeros((T, 33, 3))
    root = np.zeros((T, 3))
    hip_height = np.zeros(T)
    valid = np.zeros(T, dtype=bool)
    for t, row in enumerate(assigned):
        det = row[k]
        if det is None:
            continue
        pos = camera_position(det, width, height, fov)
        if pos is None:
            continue
        w = retarget.mediapipe_to_mmd(det["world"])
        world[t] = w
        root[t] = pos
        hip_height[t] = -w[FEET, 1].min()  # world origin is the hip centre
        valid[t] = True
    return world, root, hip_height, valid


def main(argv=None):
    args = build_args().parse_args(argv)
    base = args.output or os.path.splitext(args.video)[0]
    frames, times, info = load_or_extract(args, base + ".landmarks.pkl")
    fps = info[0]
    if not frames:
        sys.exit("Nenhum quadro lido do vídeo.")

    assigned = tracker.assign_tracks(frames, args.persons, color_weight=args.color_weight)
    tracker.apply_swaps(assigned, args.swap)
    for k in range(args.persons):
        seen = sum(1 for row in assigned if row[k] is not None)
        print(f"Pessoa {k + 1}: detectada em {seen}/{len(assigned)} quadros")

    if args.preview:
        from mpvmd.preview import render_preview
        render_preview(args.video, base + "_preview.mp4", assigned, times, args.start)
        print(f"Preview: {base}_preview.mp4")

    people = []
    for k in range(args.persons):
        world, root, hip_height, valid = build_person_arrays(assigned, k, info, args.fov)
        if not valid.any():
            print(f"Pessoa {k + 1}: nunca detectada, nada a exportar.")
            people.append(None)
            continue
        world, _ = filters.fill_gaps(world, valid, args.max_gap)
        root, _ = filters.fill_gaps(root, valid, args.max_gap)
        hip_height, _ = filters.fill_gaps(hip_height, valid, args.max_gap)
        filled = np.ones(len(world), dtype=bool)
        if args.smooth > 0:
            world = filters.smooth_sequence(world, filled, fps, min_cutoff=1.5 / args.smooth, beta=0.3)
            root = filters.smooth_sequence(root, filled, fps, min_cutoff=0.4 / args.smooth, beta=0.05)
            hip_height = filters.smooth_sequence(hip_height, filled, fps, min_cutoff=0.8 / args.smooth, beta=0.1)
        first = int(np.flatnonzero(valid)[0])
        people.append({"world": world, "root": root, "hip_height": hip_height, "valid": valid, "first": first})

    present = [p for p in people if p is not None]
    if not present:
        sys.exit("Nenhuma pessoa detectada.")
    shared_origin = np.mean([p["root"][p["first"]] for p in present], axis=0)

    rt = retarget.Retargeter(arm_angle=args.arm_angle, wrists=not args.no_wrists, ankles=not args.no_ankles)
    rel_times = times - times[0]
    n_vmd = int(math.floor(rel_times[-1] * VMD_FPS)) + 1
    src_index = np.searchsorted(rel_times, np.arange(n_vmd) / VMD_FPS)
    src_index = np.clip(src_index, 0, len(rel_times) - 1)

    for k, person in enumerate(people):
        if person is None:
            continue
        origin = shared_origin if args.origin == "shared" else person["root"][person["first"]]
        standing = np.percentile(person["hip_height"][person["valid"]], 90)
        keys = []
        for vf, t in enumerate(src_index):
            for name, m in rt.compute(person["world"][t]).items():
                keys.append(BoneFrame(name, vf, rotation=retarget.mat_to_quat(m)))
            if not args.no_translation:
                d = person["root"][t] - origin
                pos = (d[0] * args.scale,
                       (person["hip_height"][t] - standing) * args.scale,
                       d[2] * args.depth * args.scale)
                keys.append(BoneFrame("センター", vf, position=pos))
        out = f"{base}_pessoa{k + 1}.vmd"
        write_vmd(out, keys, ik_off_bones=None if args.keep_ik else retarget.IK_BONES)
        print(f"Gerado: {out}  ({n_vmd} quadros a {VMD_FPS} fps)")


if __name__ == "__main__":
    main()
