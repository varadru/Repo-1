"""Draws each tracked person with its own colour and number, to check identities."""
import cv2

COLORS = [(60, 220, 60), (60, 120, 255), (255, 160, 40), (220, 60, 220), (40, 220, 220)]
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (0, 11), (0, 12)]


def render_preview(video_path, out_path, assigned, times, start):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    index = out_index = 0
    while out_index < len(assigned):
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        index += 1
        if t < start:
            continue
        for k, det in enumerate(assigned[out_index]):
            if det is None:
                continue
            color = COLORS[k % len(COLORS)]
            pts = det["image_px"][:, :2].astype(int)
            for a, b in EDGES:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 3)
            head = tuple(pts[0] - [0, 40])
            cv2.putText(frame, f"P{k + 1}", head, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
            cv2.putText(frame, f"P{k + 1}", head, cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f"quadro {out_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, f"quadro {out_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        out_index += 1
    cap.release()
    writer.release()
