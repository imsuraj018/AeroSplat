# AeroSplat – COLMAP Rebuild Workflow

## What changed

AeroSplat now treats COLMAP as an explicit reconstruction stage.

### Upload workflow

1. Select a disaster site.
2. Click **Add drone images** and choose **append**.
3. After upload, AeroSplat automatically starts COLMAP when at least 3 reconstruction images are available and the COLMAP result is stale/missing.
4. The interface also exposes **Generate COLMAP** so the user can start a rebuild manually.
5. COLMAP always rebuilds from the complete current reconstruction image set for the active disaster site.
6. The sparse model is published to:
   - Legacy site: `C:\aerosplat_workspace\reconstruction\colmap\sparse\0`
   - Named site: `<site>\reconstruction\colmap\sparse\0`
7. `meta.json` stores the image-set hash and COLMAP statistics.
8. The existing 3DGS model stays visible while COLMAP runs.
9. **Rebuild digital twin** runs the full COLMAP → undistortion → GraphDeco 3DGS pipeline again.

## Important behavior

COLMAP is intentionally recomputed from all current reconstruction images. This is simpler and more deterministic than trying to incrementally modify a previous sparse model.

Evidence images do not enter COLMAP.

At least 3 overlapping images are required, but practical reconstruction quality depends on overlap, viewpoint diversity, texture, and image quality.

## Useful endpoints

- `GET /api/health`
- `GET /api/state`
- `POST /api/colmap`
- `POST /api/run?iterations=7000`
- `GET /api/model.ply`
- `GET /api/model/bounds`

## Expected state after COLMAP

Example:

```text
COLMAP current · 128 registered · 82,218 points
```

and:

```text
Images       128
Registered   128
3D points    82,218
Observations 476,154
Reprojection ~0.51 px
```

These numbers are examples from the existing South Building reconstruction, not guaranteed values for a new dataset.
