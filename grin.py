import numpy as np
import grin_utils as utils

from tqdm import tqdm
from scipy.interpolate import interp1d
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import cv2 as cv

class lens():
    def __init__(self, lattice="gyroid", eps_mat = 2.4, cell_size = 5.0, grid_size = 64, min_thickness = 0.15,
                 edge_mode='extend'):
        
        """
        Class for graded-index (GRIN) structures.

        Initializes a cubic grid representing one unit cell of a periodic
        lattice and preallocates a distance field. Grid coordinates are
        sampled on node points (edges), so each axis has `grid_size + 1`
        samples. 

        Parameters
        ----------
        lattice : {'gyroid', 'diamond'}, default 'gyroid'
            Name of the analytic lattice to use.
        eps_mat: float, default 2.35
            Dielectric constant of the printed material. Must be > 1.
        cell_size : float, default 5.0
            Physical size of the unit cell in millimeters. Must be in [1.0, 8.0].
        grid_size : int, default 64
            Number of intervals per axis. The internal grid uses `grid_size + 1`
            samples per axis.
        min_thickness : float, default 0.1
            Minimum beam thickness in mm
        """

        # Error handling
        if type(lattice) is not str:
            raise TypeError("[lattice] must be a string.")
        if lattice not in ["diamond", "gyroid", "octet", "fluorite"]:
            raise NotImplementedError(f"Selected lattice [{lattice}] is not implemented.")
        if cell_size<1.0 or cell_size>8.0:
            raise ValueError("cell_size must be between 1-8 mm")
        if edge_mode not in ['clip', 'extend']:
            raise ValueError("Edge mode should be clip or extend.")

        # Each cell is divided into grid_size elements
        self.__grid_size = grid_size + 1 # + 1 since these are edges
        self.__lattice = lattice.lower() 
        self.__cell_size = cell_size # in mm

        self.edge_mode = edge_mode

        self.unit_grid = np.ones((self.__grid_size, 
                                    self.__grid_size, 
                                    self.__grid_size))*1e6

        self.__dx = self.__cell_size / (self.__grid_size - 1) 
        self.__voxel_size = self.__dx 

        self.eps_mat = eps_mat
        self.min_thickness = min_thickness

        # Make 3D distance field
        self.__setup_beams()

    def __setup_beams(self):

        if self.__lattice == "gyroid":
            self.__make_gyroid_cell()
        elif self.__lattice == "diamond":
            self.__make_diamond_cell()
        elif self.__lattice == "octet":
            self.__make_octet_cell()
        elif self.__lattice == "fluorite":
            self.__make_fluorite_cell()
        else:
            raise NotImplementedError(f"Lattice {self.__lattice} not implemented")
        
    def __construct_thickness(self):

        self.__compute_d2t()

        density_interp_array = np.arange(0, 1.001, 0.001)
        f_d2t = interp1d(density_interp_array, self.d2t)
        f_t2d = interp1d(self.d2t, density_interp_array)

        self.thickness_grid = f_d2t(self.density_grid)
        
        # self.thickness_grid[(self.thickness_grid>0)*(self.thickness_grid<self.min_thickness)] = self.min_thickness
        if self.edge_mode == 'extend':
            self.thickness_grid[(self.thickness_grid<self.min_thickness)*(self.thickness_grid>0)] = self.min_thickness
        else:
            self.thickness_grid[(self.thickness_grid<self.min_thickness)] = 0.0

        self.density_grid_output = f_t2d(self.thickness_grid)
    
    def make(self, shape="luneburg_sphere", 
             out_shape=(101, 101, 101), R=None, X=60.0, Y=60.0, Z=0.0, 
             custom_eps_func=None, custom_eps_grid=None, eps_func_type="grid", lattice_px=1000, apply_edge_correction=False):
        
        self.__X = X
        self.__Y = Y
        self.__Z = Z

        if shape=="luneburg_sphere" and R is None:
            raise Exception("Plase set a radius for Luneburg lens!")
        self.__R = R

        self.shape = shape
        self.mask = None

        if len(out_shape) != 3:
            raise Exception("out_shape must be a tuple of 3 integers")
        self.custom_eps_func = custom_eps_func
        self.custom_eps_grid = custom_eps_grid

        self.__Nx, self.__Ny, self.__Nz = int(out_shape[0]), int(out_shape[1]), int(out_shape[2])

        self.eps_grid = np.zeros((self.__Nx, self.__Ny, self.__Nz))
        self.density_grid = np.zeros((self.__Nx, self.__Ny, self.__Nz))

        # Special case for Luneburg sphere
        if self.shape.lower() == 'luneburg_sphere':
            self.__stepx = 2*self.__R/(self.__Nx-1)
            self.__stepy = 2*self.__R/(self.__Ny-1)
            self.__stepz = 2*self.__R/(self.__Nz-1)
            self.__make_luneburg_sphere(min_eps=1.0)
            self.__construct_thickness()
            self.eps_grid_output = utils.get_eps_from_vf(self.density_grid_output, self.eps_mat)
            min_eps = np.min(self.eps_grid_output[self.eps_grid_output>1])
            self.__make_luneburg_sphere(min_eps)
            self.__construct_thickness()
            self.eps_grid_output = utils.get_eps_from_vf(self.density_grid_output, self.eps_mat)

        # Special case for Luneburg cylinder
        if self.shape.lower() == 'luneburg_cylinder':
            self.__stepx = 2*self.__R/(self.__Nx-1)
            self.__stepy = 2*self.__R/(self.__Ny-1)
            self.__stepz = self.__Z/(self.__Nz-1)
            self.__make_luneburg_cylinder()
            self.__construct_thickness()
            self.eps_grid_output = utils.get_eps_from_vf(self.density_grid_output, self.eps_mat)

        # Custom grid
        if self.shape.lower() == "custom_grid":

            if self.custom_eps_func is None and self.custom_eps_grid is None:
                raise TypeError("Custom dielectric function or grid must be specified.")
            if self.custom_eps_grid is not None:
                in_shape = np.asarray(self.custom_eps_grid).shape
                if len(in_shape) != 3:
                    raise ValueError("custom_eps_grid must be a 3D array.")
                if any(o < i for o, i in zip((self.__Nx, self.__Ny, self.__Nz), in_shape)):
                    raise ValueError(
                        "out_shape must be >= custom_eps_grid.shape in every dimension. "
                        f"Got out_shape={(self.__Nx, self.__Ny, self.__Nz)}, "
                        f"custom_eps_grid.shape={in_shape}."
                    )
            
            self.__stepx = self.__X/(self.__Nx-1)
            self.__stepy = self.__Y/(self.__Ny-1)
            self.__stepz = self.__Z/(self.__Nz-1)

            self.__make_custom_grid()
            self.__construct_thickness()
            self.eps_grid_output = utils.get_eps_from_vf(self.density_grid_output, self.eps_mat)


        
        if apply_edge_correction:
            self.apply_edgecorrection()


        # Draw lattice for visualization
        if self.__Z > 0:
            zs = np.linspace(0, self.__Z, self.__Nz)
        else:
            zs = np.linspace(0, 2*self.__R, self.__Nz)


        self.lattice_px = int(lattice_px)
        self.lattice = np.zeros((self.lattice_px, self.lattice_px, self.__Nz))
        self.__slice_cache = self.__prepare_slice_cache(self.lattice_px, self.lattice_px)
        self.__thickness_lattice = np.empty((self.lattice_px, self.lattice_px, self.__Nz), dtype=np.float64)
        if self.mask is None:
            self.mask = np.ones_like(self.thickness_grid)
        for i in range(self.__Nz):
            self.__thickness_lattice[:, :, i] = resize(
                self.thickness_grid[:, :, i]*self.mask[:, :, i],
                (self.lattice_px, self.lattice_px),
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            )

        for i in range(self.__Nz):
            self.lattice[:, :, i] = self.__make_slices(zs[i], i, self.lattice_px, self.lattice_px)

    
    def write_thickness(self, file_name, origin=None):

        if origin == None:
            origin = (-self.__X/2, -self.__Y/2, -self.__Z/2)
        with open(file_name, 'w') as f:
            
            f.write("origin \n")
            f.write(f"{origin[0]} {origin[1]} {origin[2]} \n")
            f.write(f"voxelSize \n")
            f.write(f"{round(self.__stepx, 4)} {round(self.__stepy, 4)} {round(self.__stepz, 4)} \n")
            f.write("grid \n")
            
            f.write(f"{self.__Nx} {self.__Ny} {self.__Nz} \n")
            for z in range(self.__Nz):
                for y in range(self.__Ny):
                    for x in range(self.__Nx):
                        thickness = self.thickness_grid[x,y,z]
                        f.write("{:.3f} ".format(thickness))
                    f.write("\n")

    def make_mesh(self):
        if self.shape.lower() == 'luneburg_sphere':
            self.mesh = utils.sphere_mesh(radius=self.__R)
        elif self.shape.lower() == 'luneburg_cylinder':
            self.mesh = utils.cylinder_mesh(radius=self.__R, height=self.__Z)
        else:
            self.mesh = None

    def __make_gyroid_cell(self):
        """
        Make a standard gyroid cell.
        Should not be called by user.
        """
        max_val = 1.3

        dist_scale = (0.32 / max_val) * self.__cell_size

        stepx = (1/(self.__grid_size  - 1)) * 2*np.pi
        stepy = (1/(self.__grid_size  - 1)) * 2*np.pi
        stepz = (1/(self.__grid_size  - 1)) * 2*np.pi

        fx = np.arange(self.__grid_size) * stepx
        fy = np.arange(self.__grid_size) * stepy
        fz = np.arange(self.__grid_size) * stepz

        # classic gyroid
        dist_val = (
            np.sin(fx)[:, None, None] * np.cos(fy)[None, :, None]
            + np.sin(fy)[None, :, None] * np.cos(fz)[None, None, :]
            + np.sin(fz)[None, None, :] * np.cos(fx)[:, None, None]
        )


        self.unit_grid[...] = dist_scale * np.abs(dist_val)

    def __make_diamond_cell(self):
        """
        Make a standard diamond cell.
        Should not be called by user.
        """
        beams = np.loadtxt("./beams/diamond_beams.csv", delimiter=",")
        self.__draw_beams(beams)

    def __make_octet_cell(self):
        """
        Make a standard octet cell.
        Should not be called by user.
        """
        beams = np.loadtxt("./beams/octet_beams.csv", delimiter=",")
        self.__draw_beams(beams)

    def __make_fluorite_cell(self):
        f_verts  = np.loadtxt("./beams/fluorite_fverts.csv",  delimiter=',')[:, :3]
        ca_verts = np.loadtxt("./beams/fluorite_caverts.csv", delimiter=',')[:, :3]

        k = 4  # Ca neighbors per F in this representation
        beams = []

        for va in f_verts:
            d = np.linalg.norm(ca_verts - va, axis=1)
            nn = np.argsort(d)[:k]
            for j in nn:
                vb = ca_verts[j]
                beams.append(np.hstack([va, vb]))

        beams = np.asarray(beams)  # shape (8*k, 6) = (32,6)
        self.__draw_beams(beams)



    def __draw_beam_dist(self, p0, p1, block=32):
        """
        Update 3D distance field with the distance to a line segment (beam).

        For each voxel at index (x, y, z), computes the Euclidean distance to
        the segment [p0, p1] in world units and writes:
            unit_grid[x, y, z] = min(unit_grid[x, y, z], distance)

        Coordinates of voxel centers are assumed to be:
            voxel_world = voxel_size * [x, y, z]

        Parameters
        ----------
        p0 : array_like, shape (3,)
            Segment start in world units (same units as `voxel_size`).
        p1 : array_like, shape (3,)
            Segment end in world units.
        block : int, optional
            Tile size per axis for chunked processing to bound memory. Default 32.

        Side Effects
        ------------
        Modifies `self.__unit_grid` in place. No array is returned.

        """
 
        u = p1 - p0
        ulen2 = np.dot(u, u)

        for x0 in range(0, self.__grid_size, block):
            x1 = min(x0 + block, self.__grid_size)
            xs = self.__voxel_size * np.arange(x0, x1)
            WX = xs[:, None, None] - p0[0]

            for y0 in range(0, self.__grid_size, block):
                y1 = min(y0 + block, self.__grid_size)
                ys = self.__voxel_size * np.arange(y0, y1)
                WY = ys[None, :, None] - p0[1]

                for z0 in range(0, self.__grid_size, block):
                    z1 = min(z0 + block, self.__grid_size)
                    zs = self.__voxel_size * np.arange(z0, z1)
                    WZ = zs[None, None, :] - p0[2]

                    dot_wu = WX * u[0] + WY * u[1] + WZ * u[2]
                    t = np.clip(dot_wu / ulen2, 0.0, 1.0)

                    DX = WX - t * u[0]
                    DY = WY - t * u[1]
                    DZ = WZ - t * u[2]
                    dist = np.sqrt(DX * DX + DY * DY + DZ * DZ)

                    sg = self.unit_grid[x0:x1, y0:y1, z0:z1]
                    np.minimum(sg, dist, out=sg)

    def __draw_beams(self, beams):
        """
        Compute distance between all points and each beam.
        Should not be called by the user.
        """
        for beam in beams:
            p0 = np.array([beam[0], beam[1], beam[2]])
            p1 = np.array([beam[3], beam[4], beam[5]])

            p0mm = p0 * self.__cell_size
            p1mm = p1 * self.__cell_size

            self.__draw_beam_dist(p0mm, p1mm, block=32)

    def __compute_d2t(self):
        """
        Compute density-to-thickness array via linear interpolation.
        This is lattice-dependent.
        Should not be called by the user.
        """
        d2t_array = np.loadtxt(f"./d2t/{self.__lattice}_d2t.csv", delimiter=',')
        self.d2t = utils.predict_m_at_n(np.array([1,2,3,4,5,6,7,8]), 
                                   d2t_array, 
                                   self.__cell_size, 
                                   kind='linear', 
                                   extrapolate=True)

    
    def __make_luneburg_sphere(self, min_eps = 1.0):
        midx, midy, midz = self.__Nx//2, self.__Ny//2, self.__Nz//2
        # Build coordinate grids
        x = (np.arange(self.__Nx) - midx) * self.__stepx 
        y = (np.arange(self.__Ny) - midy) * self.__stepy 
        z = (np.arange(self.__Nz) - midz) * self.__stepz 
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        r = np.sqrt(X**2 + Y**2 + Z**2)

        mask = r <= self.__R
        self.mask = mask

        eps_vals = np.maximum(utils.get_lun_eps(r[mask], self.__R, min_eps), 1.0)
        self.eps_grid[mask] = eps_vals
        self.eps_grid[~mask] = 1.0
        self.density_grid[mask] = np.clip(utils.get_vf(eps_vals, self.eps_mat), 0.0, 1.0)
        self.density_grid[~mask] = 0.0

        

    def __make_luneburg_cylinder(self):
        midx, midy, midz = self.__Nx//2, self.__Ny//2, self.__Nz//2
        # Build coordinate grids
        x = (np.arange(self.__Nx) - midx) * self.__stepx
        y = (np.arange(self.__Ny) - midy) * self.__stepy
        z = (np.arange(self.__Nz) - midz) * self.__stepz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        r = np.sqrt(X**2 + Y**2)

        mask = r <= self.__R

        eps_vals = np.maximum(utils.get_lun_eps(r[mask], self.__R), 1.0)
        self.eps_grid[mask] = eps_vals
        self.eps_grid[~mask] = 1.0
        self.density_grid[mask] = np.clip(utils.get_vf(eps_vals, self.eps_mat), 0.0, 1.0)
        self.density_grid[~mask] = 0.0

    def __make_custom_grid(self):
        x = (np.arange(self.__Nx)) * self.__stepx
        y = (np.arange(self.__Ny)) * self.__stepy
        z = (np.arange(self.__Nz)) * self.__stepz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        if self.custom_eps_grid is None:
            eps_vals = np.maximum(self.custom_eps_func(X, Y, Z), 1.0)
            self.eps_grid = eps_vals
        else:
            eps_in = np.asarray(self.custom_eps_grid, dtype=np.float64)
            target_shape = (self.__Nx, self.__Ny, self.__Nz)
            if eps_in.shape == target_shape:
                self.eps_grid = eps_in.copy()
            else:
                # Resample to out_shape using trilinear interpolation.
                self.eps_grid = resize(
                    eps_in,
                    target_shape,
                    order=1,
                    anti_aliasing=False,
                    preserve_range=True,
                    mode="edge",
                ).astype(np.float64, copy=False)

        self.density_grid = np.clip(utils.get_vf(self.eps_grid, self.eps_mat), 0.0, 1.0)
  
                
    def __prepare_slice_cache(self, num_xpix, num_ypix):
        imgx, imgy = num_xpix, num_ypix
        img_stepx = self.__X / imgx
        img_stepy = self.__Y / imgy

        x_mm = (np.arange(imgx) + 0.5) * img_stepx
        y_mm = (np.arange(imgy) + 0.5) * img_stepy
        X, Y = np.meshgrid(x_mm, y_mm, indexing="ij")

        cell_idx_x = np.floor(X / self.__cell_size)
        cell_idx_y = np.floor(Y / self.__cell_size)

        cellx = X - cell_idx_x * self.__cell_size
        celly = Y - cell_idx_y * self.__cell_size

        i = np.floor(cellx / self.__voxel_size).astype(np.int64)
        j = np.floor(celly / self.__voxel_size).astype(np.int64)

        valid = (i >= 0) & (j >= 0) & (i < self.__grid_size) & (j < self.__grid_size)
        return i, j, valid

    def __make_slices(self, z, z_idx_out, num_xpix, num_ypix):
        
        Imgx, Imgy = num_xpix, num_ypix

        cell_size = self.__cell_size
        voxel = self.__voxel_size
        grid_size = self.__grid_size

        # k from z
        cell_idx_z = np.floor(z / cell_size)
        unitz = z - cell_size * cell_idx_z
        k = int(np.floor(unitz / voxel))
        if k < 0 or k >= grid_size:
            return np.zeros((Imgx, Imgy), dtype=np.uint8)

        if self.__slice_cache is not None and self.__slice_cache[0].shape == (Imgx, Imgy):
            i, j, valid = self.__slice_cache
        else:
            i, j, valid = self.__prepare_slice_cache(Imgx, Imgy)
       

        if np.any(valid):
            d_slice = self.unit_grid[:, :, k]
            d_valid = d_slice[i[valid], j[valid]]

            if self.__thickness_lattice is not None and self.__thickness_lattice.shape[:2] == (Imgx, Imgy):
                thickness = self.__thickness_lattice[:, :, z_idx_out]
            else:
              
                thickness = resize(
                    self.thickness_grid[:, :, z_idx_out],
                    (Imgx, Imgy),
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True,
                )
              

            out = np.zeros((Imgx, Imgy), dtype=np.uint8)
            hit = np.zeros_like(valid)
            hit[valid] = d_valid < thickness[valid]
            out[hit] = 1
            
            return out
       

        return np.zeros((Imgx, Imgy), dtype=np.uint8)
    

    def apply_edgecorrection(self):
        """Applies a dilation to the 3D grids to prevent edge artifacts when 3D printing.
        """

        for islice in range(len(self.thickness_grid)):
        
            kernel = np.ones((3,3),np.uint8)

            eps = self.eps_grid[:,:,islice].copy()
            epsoutput = self.eps_grid_output[:,:,islice].copy()
            thick = self.thickness_grid[:,:,islice].copy()
            fill = self.density_grid[:,:,islice].copy()

            # inside mesh?
            inside_mesh_mask = self.eps_grid[:,:,islice]==1


            eps_inside = self.eps_grid[:,:,islice].copy()
            eps_inside[inside_mesh_mask] = 0

            epsoutput_inside = self.eps_grid_output[:,:,islice].copy()
            epsoutput_inside[inside_mesh_mask] = 0

            thick_inside = self.thickness_grid[:,:,islice].copy()
            thick_inside[inside_mesh_mask] = 0

            fill_inside = self.density_grid[:,:,islice].copy()
            fill_inside[inside_mesh_mask] = 0


            mask_out = self.eps_grid[:,:,islice].copy()
            mask_out = mask_out==1

            # apply filters
            eps_dilation = cv.dilate(eps,kernel,iterations = 1)
            eps_dilation = gaussian_filter(eps_dilation, 1)

            epsoutput_dilation = cv.dilate(epsoutput,kernel,iterations = 1)
            epsoutput_dilation = gaussian_filter(epsoutput_dilation, 1)

            thick_dilation = cv.dilate(thick,kernel,iterations = 1)
            thick_dilation = gaussian_filter(thick_dilation, 1)

            fill_dilation = cv.dilate(fill,kernel,iterations = 1)
            fill_dilation = gaussian_filter(fill_dilation, 1)

            self.eps_grid[:,:,islice] = eps_dilation*mask_out + eps_inside
            self.eps_grid_output[:,:,islice] = epsoutput_dilation*mask_out + epsoutput_inside
            self.thickness_grid[:,:,islice] = thick_dilation*mask_out + thick_inside
            self.density_grid[:,:,islice] = fill_dilation*mask_out + fill_inside

            
        

        





  
