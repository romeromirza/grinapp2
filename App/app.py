import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skimage.transform import resize

# Ensure grin.py can resolve ./beams and ./d2t regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grin  # noqa: E402

LATTICE_XY_PX = 500


def parse_grid_file(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    try:
        data = np.loadtxt(io.StringIO(text), delimiter=None, skiprows=1)
    except Exception:
        data = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1)

    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("File must contain exactly 4 float columns: x y z Dk")

    x, y, z, values = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    z_unique = np.unique(z)

    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    dk_3d = np.ones((nx, ny, nz), dtype=np.float64)

    x_map = {v: i for i, v in enumerate(x_unique)}
    y_map = {v: j for j, v in enumerate(y_unique)}
    z_map = {v: k for k, v in enumerate(z_unique)}

    for xi, yi, zi, val in zip(x, y, z, values):
        i = x_map[xi]
        j = y_map[yi]
        k = z_map[zi]
        dk_3d[i, j, k] = val

    extents = {
        "X": float(np.max(x_unique) - np.min(x_unique)),
        "Y": float(np.max(y_unique) - np.min(y_unique)),
        "Z": float(np.max(z_unique) - np.min(z_unique)),
    }
    origin = (
        -float(extents["X"]) / 2.0,
        -float(extents["Y"]) / 2.0,
        -float(extents["Z"]) / 2.0,
    )
    return dk_3d, extents, origin


def make_custom_function(fn_src):
    local_ns = {}
    safe_globals = {"np": np, "__builtins__": {}}
    exec(fn_src, safe_globals, local_ns)
    if "custom_dk" in local_ns:
        fn = local_ns["custom_dk"]
    elif "custom_eps" in local_ns:
        fn = local_ns["custom_eps"]
    else:
        raise ValueError("Function text must define: custom_dk(x, y, z)")
    if not callable(fn):
        raise ValueError("custom_dk must be callable")
    return fn


def get_slice(volume, axis, idx):
    if axis == "x":
        ii = min(idx, volume.shape[0] - 1)
        return volume[ii, :, :]
    if axis == "y":
        jj = min(idx, volume.shape[1] - 1)
        return volume[:, jj, :]
    kk = min(idx, volume.shape[2] - 1)
    return volume[:, :, kk]


def sample_lattice_slice_mm(lens_obj, axis, idx_eps, idx_lat, mm_step=0.1):
    nx, ny, nz = lens_obj.eps_grid.shape
    mm_step = max(1e-6, float(mm_step))
    x_span = float(lens_obj._lens__X)
    y_span = float(lens_obj._lens__Y)
    z_span = float(lens_obj._lens__Z) if float(lens_obj._lens__Z) > 0 else 2.0 * float(lens_obj._lens__R)

    if axis == "x":
        target_shape = (max(3, int(np.ceil(y_span / mm_step))), max(3, int(np.ceil(z_span / mm_step))))
    elif axis == "y":
        target_shape = (max(3, int(np.ceil(x_span / mm_step))), max(3, int(np.ceil(z_span / mm_step))))
    else:
        target_shape = (max(3, int(np.ceil(x_span / mm_step))), max(3, int(np.ceil(y_span / mm_step))))

    # Prefer backend lattice sampler for z-plane for highest fidelity.
    if axis == "z":
        sampler = getattr(lens_obj, "_lens__make_slices", None)
        if callable(sampler):
            z_mm = (float(idx_eps) / max(nz - 1, 1)) * z_span
            try:
                return np.asarray(sampler(z_mm, target_shape[0], target_shape[1]), dtype=np.float32)
            except Exception:
                pass

    # Fallback for x/y (and z if sampler fails): nearest-neighbor upsample of current lattice slice.
    base_slice = np.asarray(get_slice(lens_obj.lattice, axis, idx_lat), dtype=np.float32)
    return resize(
        base_slice,
        target_shape,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
        mode="edge",
    ).astype(np.float32, copy=False)


def shape_from_mm(x_span, y_span, z_span):
    mm_step = 1.0
    nx = max(3, int(np.ceil(float(x_span) / mm_step)) + 1)
    ny = max(3, int(np.ceil(float(y_span) / mm_step)) + 1)
    nz = max(3, int(np.ceil(float(z_span) / mm_step)) + 1)
    return (nx, ny, nz)


def heatmap_fig(data, title, colorscale, zmin, zmax, height=290, high_quality=False):
    zdata = np.asarray(data, dtype=np.float32).T
    # Use optional downsampling for speed on most panels; keep full-res for quality-critical views.
    downsampled = False
    if not high_quality:
        max_side = 320
        sx = max(1, int(np.ceil(zdata.shape[0] / max_side)))
        sy = max(1, int(np.ceil(zdata.shape[1] / max_side)))
        if sx > 1 or sy > 1:
            zdata = zdata[::sx, ::sy]
            downsampled = True

    # If we downsample, use Heatmap with smoothing for better visual quality.
    if hasattr(go, "Heatmapgl") and not high_quality and not downsampled:
        trace = go.Heatmapgl(
            z=zdata,
            colorscale=colorscale,
            colorbar={"title": "value"},
            zmin=zmin,
            zmax=zmax,
        )
    else:
        trace = go.Heatmap(
            z=zdata,
            colorscale=colorscale,
            colorbar={"title": "value"},
            zsmooth="best" if (high_quality or downsampled) else False,
            zmin=zmin,
            zmax=zmax,
        )
    fig = go.Figure(data=trace)
    fig.update_layout(
        title=title,
        margin={"l": 10, "r": 10, "t": 36, "b": 10},
        height=height,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#1f2937"},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)
    return fig


def get_cached_slice_fig(lens_id, grid_name, axis, idx, data, title, colorscale, zmin, zmax):
    cache = st.session_state.setdefault("slice_plot_cache", {})
    key = (lens_id, grid_name, axis, int(idx))
    if key not in cache:
        cache[key] = heatmap_fig(data, title, colorscale, zmin, zmax)
    return cache[key]


def cut_mask_x(mask, x_keep_pct):
    m = np.asarray(mask, dtype=np.uint8).copy()
    nx = m.shape[0]
    x_idx = int(np.clip(round((float(x_keep_pct) / 100.0) * (nx - 1)), 0, nx - 1))
    if x_idx < nx - 1:
        m[x_idx + 1 :, :, :] = 0
    return m, x_idx


def add_x_cut_face(fig, dk_grid, meta, x_idx):
    x_span = meta["X"] if meta["X"] > 0 else 2.0 * meta["R"]
    y_span = meta["Y"] if meta["Y"] > 0 else 2.0 * meta["R"]
    z_span = meta["Z"] if meta["Z"] > 0 else 2.0 * meta["R"]
    nx, ny, nz = dk_grid.shape
    x0, y0, z0 = -0.5 * x_span, -0.5 * y_span, -0.5 * z_span

    x_coords = np.linspace(x0, x0 + x_span, nx, dtype=np.float64)
    y_coords = np.linspace(y0, y0 + y_span, ny, dtype=np.float64)
    z_coords = np.linspace(z0, z0 + z_span, nz, dtype=np.float64)

    x_idx = int(np.clip(x_idx, 0, nx - 1))
    dk_slice = np.asarray(dk_grid[x_idx, :, :], dtype=np.float64)
    mask_slice = dk_slice > 1.0001
    if not np.any(mask_slice):
        return fig

    # Build triangles only for valid cells to avoid long protruding artifacts.
    vx, vy, vz, vi = [], [], [], []
    ti, tj, tk = [], [], []
    vmap = {}

    def add_v(j, k):
        key = (j, k)
        if key in vmap:
            return vmap[key]
        idx = len(vx)
        vmap[key] = idx
        vx.append(x_coords[x_idx])
        vy.append(y_coords[j])
        vz.append(z_coords[k])
        vi.append(float(dk_slice[j, k]))
        return idx

    for j in range(ny - 1):
        for k in range(nz - 1):
            if mask_slice[j, k] and mask_slice[j + 1, k] and mask_slice[j, k + 1] and mask_slice[j + 1, k + 1]:
                a = add_v(j, k)
                b = add_v(j + 1, k)
                c = add_v(j, k + 1)
                d = add_v(j + 1, k + 1)
                ti.extend([a, b])
                tj.extend([b, d])
                tk.extend([d, c])

    if not ti:
        return fig

    dk_min = float(np.min(dk_grid))
    dk_max = float(np.max(dk_grid))

    fig.add_trace(
        go.Mesh3d(
            x=np.asarray(vx),
            y=np.asarray(vy),
            z=np.asarray(vz),
            i=np.asarray(ti),
            j=np.asarray(tj),
            k=np.asarray(tk),
            intensity=np.asarray(vi),
            colorscale="Turbo",
            cmin=dk_min,
            cmax=dk_max,
            opacity=1.0,
            showscale=False,
            hoverinfo="none",
            flatshading=True,
        )
    )
    return fig


def sample_dk_at_points(points_xyz, dk_grid, meta):
    from scipy.interpolate import RegularGridInterpolator

    nx, ny, nz = dk_grid.shape
    x_span = meta["X"] if meta["X"] > 0 else 2.0 * meta["R"]
    y_span = meta["Y"] if meta["Y"] > 0 else 2.0 * meta["R"]
    z_span = meta["Z"] if meta["Z"] > 0 else 2.0 * meta["R"]
    x0, y0, z0 = -0.5 * x_span, -0.5 * y_span, -0.5 * z_span

    gx = np.linspace(x0, x0 + x_span, nx, dtype=np.float64)
    gy = np.linspace(y0, y0 + y_span, ny, dtype=np.float64)
    gz = np.linspace(z0, z0 + z_span, nz, dtype=np.float64)
    interp = RegularGridInterpolator((gx, gy, gz), dk_grid, bounds_error=False, fill_value=float(np.min(dk_grid)))
    return interp(points_xyz)


def mask_to_polydata(mask, meta, threshold, decimate, smooth_iterations):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    mask = (np.asarray(mask) > 0).astype(np.uint8, copy=False)
    if mask.max() == 0:
        raise ValueError("Mask is empty.")

    nx, ny, nz = mask.shape
    x_span = meta["X"] if meta["X"] > 0 else 2.0 * meta["R"]
    y_span = meta["Y"] if meta["Y"] > 0 else 2.0 * meta["R"]
    z_span = meta["Z"] if meta["Z"] > 0 else 2.0 * meta["R"]
    spacing = (
        float(x_span) / max(nx - 1, 1),
        float(y_span) / max(ny - 1, 1),
        float(z_span) / max(nz - 1, 1),
    )
    origin = (-0.5 * float(x_span), -0.5 * float(y_span), -0.5 * float(z_span))

    mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    nx, ny, nz = mask.shape
    origin = (origin[0] - spacing[0], origin[1] - spacing[1], origin[2] - spacing[2])

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)

    vtk_arr = numpy_to_vtk(mask.ravel(order="F"), deep=False, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_arr.SetName("mask")
    image.GetPointData().SetScalars(vtk_arr)

    surface = vtk.vtkFlyingEdges3D() if hasattr(vtk, "vtkFlyingEdges3D") else vtk.vtkMarchingCubes()
    if not hasattr(vtk, "vtkFlyingEdges3D"):
        surface.ComputeNormalsOn()
    surface.SetInputData(image)
    surface.SetValue(0, float(threshold))
    current_port = surface.GetOutputPort()

    if decimate > 0:
        dec = vtk.vtkDecimatePro()
        dec.SetInputConnection(current_port)
        dec.SetTargetReduction(float(min(max(decimate, 0.0), 0.95)))
        dec.PreserveTopologyOn()
        current_port = dec.GetOutputPort()

    if smooth_iterations > 0:
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(current_port)
        smooth.SetNumberOfIterations(int(smooth_iterations))
        smooth.SetRelaxationFactor(0.1)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOn()
        current_port = smooth.GetOutputPort()

    # Keep only the largest connected shell to show outer skin.
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputConnection(current_port)
    conn.SetExtractionModeToLargestRegion()
    current_port = conn.GetOutputPort()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(current_port)
    tri.Update()
    return tri.GetOutput()


def polydata_to_plotly(polydata, dk_grid, meta):
    import pyvista as pv

    mesh = pv.wrap(polydata)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("Empty mesh after extraction.")

    faces = mesh.faces.reshape(-1, 4)
    i, j, k = faces[:, 1], faces[:, 2], faces[:, 3]
    x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]

    dk_vals = sample_dk_at_points(mesh.points, dk_grid, meta)
    dk_min = float(np.min(dk_grid))
    dk_max = float(np.max(dk_grid))

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                intensity=dk_vals,
                colorscale="Turbo",
                cmin=dk_min,
                cmax=dk_max,
                colorbar={"title": "Dk"},
                opacity=1.0,
                flatshading=False,
                lighting={"ambient": 0.4, "diffuse": 0.8, "specular": 0.2, "roughness": 0.7},
            )
        ]
    )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene={
            "aspectmode": "data",
            "bgcolor": "white",
            "xaxis": {
                "visible": False,
                "showgrid": False,
                "zeroline": False,
                "showbackground": False,
                "showticklabels": False,
                "title": "",
            },
            "yaxis": {
                "visible": False,
                "showgrid": False,
                "zeroline": False,
                "showbackground": False,
                "showticklabels": False,
                "title": "",
            },
            "zaxis": {
                "visible": False,
                "showgrid": False,
                "zeroline": False,
                "showbackground": False,
                "showticklabels": False,
                "title": "",
            },
        },
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_vtk_mesh(dk_bytes, meta, threshold, decimate, smooth_iterations, x_keep_pct):
    dk_grid = np.load(io.BytesIO(dk_bytes))
    mask = (dk_grid > 1.0001).astype(np.uint8)
    cut_mask, x_idx = cut_mask_x(mask, x_keep_pct)
    polydata = mask_to_polydata(cut_mask, meta, threshold, decimate, smooth_iterations)
    fig = polydata_to_plotly(polydata, dk_grid, meta)
    if x_keep_pct < 100 and x_idx < (dk_grid.shape[0] - 1):
        fig = add_x_cut_face(fig, dk_grid, meta, x_idx)
    return fig


st.set_page_config(layout="wide", page_title="GRIN Lens Builder")
st.title("GRIN Lens Builder")
st.markdown(
    """
<style>
[data-testid="stStatusWidget"] {
    display: none !important;
}

.stApp {
    background: #ffffff;
    color: #1f2937;
}
h1, h2, h3, .stMarkdown, .stCaption, label {
    color: #1f2937 !important;
}
div[data-testid="stMetric"], div[data-testid="stExpander"] {
    background-color: #ffffff;
}
.stButton > button, .stDownloadButton > button {
    background: #ff8a3d;
    color: #ffffff;
    border: 1px solid #e56f20;
    font-weight: 700;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #ff9d5e;
    border-color: #e56f20;
}
button[data-baseweb="tab"] {
    background: #eef2f8 !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #d7dee9 !important;
    margin-right: 8px !important;
    color: #344054 !important;
    font-weight: 700 !important;
    min-height: 48px !important;
    padding: 0.7rem 1.2rem !important;
    font-size: 1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #dfe8f7 !important;
    color: #1d4e89 !important;
    border-color: #85a9d9 !important;
    box-shadow: inset 0 -4px 0 #3b82f6 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

left, center, right = st.columns([1.05, 3.9, 1.05], gap="medium")

with left:
    st.subheader("Lens Parameters")

    lattice = st.selectbox("Lattice", ["gyroid", "diamond", "octet", "fluorite"])
    eps_mat = st.number_input("Material Dk", min_value=1.01, value=2.4, step=0.01)
    cell_size = st.number_input("Cell size (mm)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
    min_thickness = st.number_input("Min thickness (mm)", min_value=0.0, value=0.15, step=0.01)
    st.caption("Minimum allowed beam radius used during thickness mapping.")
    edge_mode = "extend"
    shape_label = st.selectbox(
        "Lens shape / source",
        ["Luneburg Sphere", "Luneburg Cylinder", "Custom Grid", "Custom Function"],
    )
    shape_map = {
        "Luneburg Sphere": "luneburg_sphere",
        "Luneburg Cylinder": "luneburg_cylinder",
        "Custom Grid": "custom_grid",
        "Custom Function": "custom_function",
    }
    shape_mode = shape_map[shape_label]

    make_kwargs = {}
    write_origin = None
    custom_grid_dims_mm = None
    custom_fn_src = None
    out_shape = None

    if shape_mode == "luneburg_sphere":
        r = st.number_input("Radius R (mm)", min_value=0.1, value=20.0, step=0.5)
        x_len = 2 * float(r)
        y_len = 2 * float(r)
        z_len = 2 * float(r)
        out_shape = shape_from_mm(x_len, y_len, z_len)
        make_kwargs.update({"shape": "luneburg_sphere", "R": float(r), "X": x_len, "Y": y_len, "Z": z_len, "out_shape": out_shape, "lattice_px": LATTICE_XY_PX})
    elif shape_mode == "luneburg_cylinder":
        r = st.number_input("Radius R (mm)", min_value=0.1, value=20.0, step=0.5)
        z_len = st.number_input("Cylinder height Z (mm)", min_value=0.1, value=120.0, step=0.5)
        x_len = 2 * float(r)
        y_len = 2 * float(r)
        z_len = float(z_len)
        out_shape = shape_from_mm(x_len, y_len, z_len)
        make_kwargs.update({"shape": "luneburg_cylinder", "R": float(r), "X": x_len, "Y": y_len, "Z": z_len, "out_shape": out_shape, "lattice_px": LATTICE_XY_PX})
    elif shape_mode == "custom_function":
        x_len = st.number_input("X span (mm)", min_value=0.1, value=120.0, step=0.5)
        y_len = st.number_input("Y span (mm)", min_value=0.1, value=120.0, step=0.5)
        z_len = st.number_input("Z span (mm)", min_value=0.1, value=120.0, step=0.5)
        out_shape = shape_from_mm(x_len, y_len, z_len)
        default_fn = """def custom_dk(x, y, z):
    # x, y, z are 3D numpy arrays
    r = np.sqrt((x - x.mean())**2 + (y - y.mean())**2 + (z - z.mean())**2)
    return 1.0 + np.clip(1.0 - r / np.max(r), 0.0, 1.0)
"""
        custom_fn_src = st.text_area("Python function (define custom_dk)", value=default_fn, height=180)
        make_kwargs.update({
            "shape": "custom_grid",
            "X": float(x_len),
            "Y": float(y_len),
            "Z": float(z_len),
            "out_shape": out_shape,
            "lattice_px": LATTICE_XY_PX,
        })
    else:
        grid_units = st.selectbox("Grid coordinate units", ["mm", "cm", "m"], index=0)
        unit_to_mm = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
        scale_to_mm = unit_to_mm[grid_units]
        uploaded = st.file_uploader("Grid file (.txt/.tsv/.csv)", type=["txt", "tsv", "csv"])
        if uploaded is not None:
            try:
                dk_3d, extents, write_origin = parse_grid_file(uploaded)
                extents_mm = {
                    "X": float(extents["X"]) * scale_to_mm,
                    "Y": float(extents["Y"]) * scale_to_mm,
                    "Z": float(extents["Z"]) * scale_to_mm,
                }
                write_origin_mm = (
                    float(write_origin[0]) * scale_to_mm,
                    float(write_origin[1]) * scale_to_mm,
                    float(write_origin[2]) * scale_to_mm,
                )
                derived_shape = shape_from_mm(extents_mm["X"], extents_mm["Y"], extents_mm["Z"])
                out_shape = tuple(max(a, b) for a, b in zip(derived_shape, dk_3d.shape))
                custom_grid_dims_mm = extents_mm
                make_kwargs.update({
                    "shape": "custom_grid",
                    "out_shape": out_shape,
                    "X": extents_mm["X"],
                    "Y": extents_mm["Y"],
                    "Z": extents_mm["Z"],
                    "custom_eps_grid": dk_3d,
                    "lattice_px": LATTICE_XY_PX,
                })
                write_origin = write_origin_mm
            except Exception as exc:
                st.error(f"Invalid grid file: {exc}")
        else:
            st.info("Upload a 4-column text grid (x, y, z, Dk) with one header row.")

    if out_shape is not None:
        st.success(f"Output grid will have shape {out_shape[0]}, {out_shape[1]}, {out_shape[2]}. A correction has been auto-applied to edges to prevent printing defects.")

    if st.button("Make lens", type="primary"):
        try:
            with st.spinner("Generating lens..."):
                if shape_mode == "custom_function":
                    make_kwargs["custom_eps_func"] = make_custom_function(custom_fn_src)

                lens_obj = grin.lens(
                    lattice=lattice,
                    eps_mat=float(eps_mat),
                    cell_size=float(cell_size),
                    min_thickness=float(min_thickness),
                    edge_mode=edge_mode,
                )
                lens_obj.make(**make_kwargs)
            st.session_state["lens"] = lens_obj
            st.session_state["shape_mode"] = shape_mode
            st.session_state["write_origin"] = write_origin
            st.session_state["custom_grid_dims_mm"] = custom_grid_dims_mm
            st.session_state["lens_id"] = int(st.session_state.get("lens_id", 0)) + 1
            st.session_state["slice_plot_cache"] = {}
            st.session_state.pop("sampled_lattice_slice", None)
            st.info("Lens ready.")
        except Exception as exc:
            st.error(f"Failed to make lens: {exc}")

with center:
    lens_ready = st.session_state.get("lens")
    if lens_ready is not None:
        st.markdown("---")
        lens_id = int(st.session_state.get("lens_id", 0))
        axis = st.selectbox("Slice axis", ["x", "y", "z"], index=2)
        ax_map = {"x": 0, "y": 1, "z": 2}
        pos = st.slider("Slice position (%)", min_value=0, max_value=100, value=50)
        axis_idx = ax_map[axis]

        eps_len = lens_ready.eps_grid.shape[axis_idx]
        lattice_len = lens_ready.lattice.shape[axis_idx]
        idx_eps = int(round((pos / 100.0) * max(eps_len - 1, 1)))
        idx_lat = int(round((pos / 100.0) * max(lattice_len - 1, 1)))

        dk_min, dk_max = float(np.min(lens_ready.eps_grid_output)), float(np.max(lens_ready.eps_grid_output))
        thick_min, thick_max = float(np.min(lens_ready.thickness_grid)), float(np.max(lens_ready.thickness_grid))
        fill_min, fill_max = float(np.min(lens_ready.density_grid)), float(np.max(lens_ready.density_grid))

        p1, p2, p3 = st.columns(3)
        with p1:
            dk_slice = get_slice(lens_ready.eps_grid_output, axis, idx_eps)
            st.plotly_chart(
                get_cached_slice_fig(
                    lens_id,
                    "dk",
                    axis,
                    idx_eps,
                    dk_slice,
                    "Dk distribution",
                    "Viridis",
                    dk_min,
                    dk_max,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with p2:
            thick_slice = get_slice(lens_ready.thickness_grid, axis, idx_eps)
            st.plotly_chart(
                get_cached_slice_fig(
                    lens_id,
                    "thickness",
                    axis,
                    idx_eps,
                    thick_slice,
                    "Beam thickness (mm)",
                    "Cividis",
                    thick_min,
                    thick_max,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with p3:
            fill_slice = get_slice(lens_ready.density_grid, axis, idx_eps)
            st.plotly_chart(
                get_cached_slice_fig(
                    lens_id,
                    "fill",
                    axis,
                    idx_eps,
                    fill_slice,
                    "Solid fill fraction",
                    "Plasma",
                    fill_min,
                    fill_max,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        lattice_slice = get_slice(lens_ready.lattice, axis, idx_lat)
        lattice_fig = heatmap_fig(
            lattice_slice,
            "Lattice",
            "Greys",
            0.0,
            1.0,
            height=560,
            high_quality=True,
        )
        lattice_fig.update_traces(showscale=False)
        lattice_fig.update_layout(title=f"Lattice ({axis}-axis, {idx_lat + 1}/{lattice_len})")
        st.plotly_chart(
            lattice_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Make a lens to view slice plots.")

with right:
    st.subheader("Exports")
    lens_ready = st.session_state.get("lens")
    if lens_ready is None:
        st.caption("Make a lens to enable export options.")
    else:
        shape_mode = st.session_state.get("shape_mode", "")

        if shape_mode == "luneburg_sphere":
            st.markdown("**STL Export (Sphere only)**")
            stl_name = st.text_input("STL file name", value="lens.stl")
            if not stl_name.lower().endswith(".stl"):
                stl_name = f"{stl_name}.stl"
            try:
                lens_ready.make_mesh()
                stl_bytes = lens_ready.mesh.export(file_type="stl")
                if isinstance(stl_bytes, str):
                    stl_bytes = stl_bytes.encode("utf-8")
                st.download_button("Download STL", data=stl_bytes, file_name=stl_name, mime="model/stl")
            except Exception as exc:
                st.error(f"STL export failed: {exc}")
        else:
            st.caption("STL export is only enabled for luneburg sphere.")

        st.markdown("**Thickness Export**")
        ib_name = st.text_input("Thickness file name", value="lens.ibgrid3d")
        if not ib_name.lower().endswith(".ibgrid3d"):
            ib_name = f"{ib_name}.ibgrid3d"

        custom_dims = st.session_state.get("custom_grid_dims_mm")
        if shape_mode == "custom_grid":
            if custom_dims is None:
                custom_dims = {
                    "X": float(lens_ready._lens__X),
                    "Y": float(lens_ready._lens__Y),
                    "Z": float(lens_ready._lens__Z),
                }
            st.caption(
                f"Inferred custom grid dimensions (mm): "
                f"{custom_dims['X']:.3f}, {custom_dims['Y']:.3f}, {custom_dims['Z']:.3f}"
            )
            default_origin = (
                -float(custom_dims["X"]) / 2.0,
                -float(custom_dims["Y"]) / 2.0,
                0.0,
            )
        elif shape_mode == "luneburg_sphere":
            radius = float(lens_ready._lens__R)
            default_origin = (-radius, -radius, -radius)
        else:
            default_origin = (0.0, 0.0, 0.0)

        st.markdown("**Thickness Origin (mm)**")
        st.caption("Reference coordinate of the output thickness grid origin in millimeters (x, y, z).")
        o1, o2, o3 = st.columns(3)
        with o1:
            origin_x = st.number_input("X", value=float(default_origin[0]), step=1.0, format="%.4f")
        with o2:
            origin_y = st.number_input("Y", value=float(default_origin[1]), step=1.0, format="%.4f")
        with o3:
            origin_z = st.number_input("Z", value=float(default_origin[2]), step=1.0, format="%.4f")

        thickness_bytes = None
        thickness_err = None
        try:
            with st.spinner("Preparing thickness export..."):
                with tempfile.TemporaryDirectory() as td:
                    out_path = Path(td) / ib_name
                    origin = (float(origin_x), float(origin_y), float(origin_z))
                    lens_ready.write_thickness(str(out_path), origin=origin)
                    thickness_bytes = out_path.read_bytes()
        except Exception as exc:
            thickness_err = str(exc)

        if thickness_bytes is not None:
            st.download_button(
                "Export thickness file (.ibgrid3d)",
                data=thickness_bytes,
                file_name=ib_name,
                mime="application/octet-stream",
                use_container_width=True,
                key="download_thickness_ibgrid3d",
            )
        else:
            st.error(f"Thickness export failed: {thickness_err}")
