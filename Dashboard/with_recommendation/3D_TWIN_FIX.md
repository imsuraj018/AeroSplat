# 3D Digital Twin fix

The server now selects Gaussian models per scene instead of searching all jobs/tests by modification time.
For the legacy scene it prefers the highest valid iteration under `C:\\aerosplat_workspace\\gaussian_output\\point_cloud\\iteration_*` and only falls back to `gaussian_test` when the authoritative output has no model.

The browser viewer uses the local ES-module build, non-progressive PLY loading, `showLoadingUI:false`, and spherical-harmonics degree 2.

After replacing files, restart the server and hard-refresh the browser. Check `/api/state`, `/api/model/info`, and `/api/model.ply`.
