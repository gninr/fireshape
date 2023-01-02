from .innerproduct import InnerProduct
import ROL
import firedrake as fd

__all__ = ["FeControlSpace", "FeMultiGridControlSpace", "BsplineControlSpace",
           "WaveletControlSpace", "ControlVector"]

# new imports for splines
from firedrake.petsc import PETSc
from functools import reduce
from itertools import product
from scipy.interpolate import splev, BSpline
from scipy.special import binom
from math import ceil, floor, log2, factorial
import numpy as np
import matplotlib.pyplot as plt


class ControlSpace(object):
    """
    ControlSpace is the space of geometric transformations.

    A transformation is identified with a domain using Firedrake.
    In particular, a transformation is converted into a domain by
    interpolating it on a Firedrake Lagrangian finite element space.

    Notational convention:
        self.mesh_r is the initial physical mesh (reference domain)
        self.V_r is a Firedrake vectorial Lagrangian finite element
            space on mesh_r
        self.id is the element of V_r that satisfies id(x) = x for every x
        self.T is the interpolant of a ControlSpace variable in self.V_r
        self.mesh_m is the mesh that corresponds to self.T (moved domain)
        self.V_m is the Firedrake vectorial Lagrangian finite element
            space on mesh_m
        self.inner_product is the inner product of the ControlSpace

    Key idea: solve state and adjoint equations on mesh_m. Then, evaluate
    shape derivatives along directions in V_m, transplant this to V_r,
    restrict to ControlSpace, compute the update (using inner_product),
    and interpolate it into V_r to update mesh_m.

    Note: transplant to V_r means creating an function in V_r and using the
    values of the directional shape derivative as coefficients. Since
    Lagrangian finite elements are parametric, no transformation matrix is
    required by this operation.
    """

    def restrict(self, residual, out):
        """
        Restrict from self.V_r into ControlSpace

        Input:
        residual: fd.Function, is a variable in the dual of self.V_r
        out: ControlVector, is a variable in the dual of ControlSpace
             (overwritten with result)
        """

        raise NotImplementedError

    def interpolate(self, vector, out):
        """
        Interpolate from ControlSpace into self.V_r

        Input:
        vector: ControlVector, is a variable in ControlSpace
        out: fd.Function, is a variable in self.V_r, is overwritten with
             the result
        """

        raise NotImplementedError

    def update_domain(self, q: 'ControlVector'):
        """
        Update the interpolant self.T with q
        """

        # Check if the new control is different from the last one.  ROL is
        # sometimes a bit strange in that it calls update on the same value
        # more than once, in that case we don't want to solve the PDE over
        # again.

        if not hasattr(self, 'lastq') or self.lastq is None:
            self.lastq = q.clone()
            self.lastq.set(q)
        else:
            self.lastq.axpy(-1., q)
            # calculate l2 norm (faster)
            diff = self.lastq.vec_ro().norm()
            self.lastq.axpy(+1., q)
            if diff < 1e-20:
                return False
            else:
                self.lastq.set(q)
        q.to_coordinatefield(self.T)
        self.T += self.id
        return True

    def get_zero_vec(self):
        """
        Create the object that stores the data for a ControlVector.

        It returns a fd.Function or a PETSc.Vec.
        It is only used in ControlVector.__init__.
        """

        raise NotImplementedError

    def assign_inner_product(self, inner_product):
        """
        create self.inner_product
        """
        raise NotImplementedError

    def get_space_for_inner(self):
        """
        Return the functionspace V to define the inner product on
        and possibly an interpolation matrix I between the finite element
        functions in V and the control functions. Note that this matrix
        is not necessarily related to self.restict() and self.interpolate()
        """
        raise NotImplementedError

    def store(self, vec, filename):
        """
        Store the vector to a file to be reused in a later computation
        """
        raise NotImplementedError

    def load(self, vec, filename):
        """
        Load a vector from a file
        """
        raise NotImplementedError


class FeControlSpace(ControlSpace):
    """Use self.V_r as actual ControlSpace."""

    def __init__(self, mesh_r):
        # Create mesh_r and V_r
        self.mesh_r = mesh_r
        element = self.mesh_r.coordinates.function_space().ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)

        # Create self.id and self.T, self.mesh_m, and self.V_m.
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.interpolate(X, self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.is_DG = False

        """
        ControlSpace for discontinuous coordinate fields
        (e.g.  periodic domains)

        In Firedrake, periodic meshes are implemented using a discontinuous
        field. This implies that self.V_r contains discontinuous functions.
        To ensure domain updates do not create holes in the domain,
        use a continuous subspace self.V_c of self.V_r as control space.
        """
        if element.family() == 'Discontinuous Lagrange':
            self.is_DG = True
            self.V_c = fd.VectorFunctionSpace(self.mesh_r,
                                              "CG", element._degree)
            self.Ip = fd.Interpolator(fd.TestFunction(self.V_c),
                                      self.V_r).callable().handle

    def restrict(self, residual, out):
        if self.is_DG:
            with residual.dat.vec as w:
                self.Ip.multTranspose(w, out.vec_wo())
        else:
            with residual.dat.vec as vecres:
                with out.fun.dat.vec as vecout:
                    vecres.copy(vecout)

    def interpolate(self, vector, out):
        if self.is_DG:
            with out.dat.vec as w:
                self.Ip.mult(vector.vec_ro(), w)
        else:
            out.assign(vector.fun)

    def get_zero_vec(self):
        if self.is_DG:
            fun = fd.Function(self.V_c)
        else:
            fun = fd.Function(self.V_r)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        if self.is_DG:
            return (self.V_c, None)
        return (self.V_r, None)

    def store(self, vec, filename="control"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_CREATE) as chk:
            chk.store(vec.fun, name=filename)

    def load(self, vec, filename="control"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_READ) as chk:
            chk.load(vec.fun, name=filename)


class FeMultiGridControlSpace(ControlSpace):
    """
    FEControlSpace on given mesh and StateSpace on uniformly refined mesh.

    Use the provided mesh to construct a Lagrangian finite element control
    space. Then, refine the mesh `refinements`-times to construct
    representatives of ControlVectors that are compatible with the state
    space.

    Inputs:
        refinements: type int, number of uniform refinements to perform
                     to obtain the StateSpace mesh.
        order: type int, order of Lagrange basis functions of ControlSpace.

    Note: as of 04.03.2018, 3D is not supported by fd.MeshHierarchy.
    """

    def __init__(self, mesh_r, refinements=1, order=1):
        mh = fd.MeshHierarchy(mesh_r, refinements)
        self.mesh_hierarchy = mh

        # Control space on coarsest mesh
        self.mesh_r_coarse = self.mesh_hierarchy[0]
        self.V_r_coarse = fd.VectorFunctionSpace(self.mesh_r_coarse, "CG",
                                                 order)

        # Create self.id and self.T on refined mesh.
        element = self.V_r_coarse.ufl_element()

        self.intermediate_Ts = []
        for i in range(refinements - 1):
            mesh = self.mesh_hierarchy[i + 1]
            V = fd.FunctionSpace(mesh, element)
            self.intermediate_Ts.append(fd.Function(V))

        self.mesh_r = self.mesh_hierarchy[-1]
        element = self.V_r_coarse.ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)

        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

    def restrict(self, residual, out):
        Tf = residual
        for Tinter in reversed(self.intermediate_Ts):
            fd.restrict(Tf, Tinter)
            Tf = Tinter
        fd.restrict(Tf, out.fun)

    def interpolate(self, vector, out):
        Tc = vector.fun
        for Tinter in self.intermediate_Ts:
            fd.prolong(Tc, Tinter)
            Tc = Tinter
        fd.prolong(Tc, out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        return (self.V_r_coarse, None)

    def store(self, vec, filename="control"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_CREATE) as chk:
            chk.store(vec.fun, name=filename)

    def load(self, vec, filename="control"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_READ) as chk:
            chk.load(vec.fun, name=filename)


class BsplineControlSpace(ControlSpace):
    """ConstrolSpace based on cartesian tensorized Bsplines."""

    def __init__(self, mesh, bbox, orders, levels, fixed_dims=[],
                 boundary_regularities=None):
        """
        bbox: a list of tuples describing [(xmin, xmax), (ymin, ymax), ...]
              of a Cartesian grid that extends around the shape to be
              optimised
        orders: describe the orders (one integer per geometric dimension)
                of the tensor-product B-spline basis. A univariate B-spline
                has order "o" if it is a piecewise polynomial of degree
                "o-1". For instance, a hat function is a B-spline of
                order 2 and thus degree 1.
        levels: describe the subdivision levels (one integers per
                geometric dimension) used to construct the knots of
                univariate B-splines
        fixed_dims: dimensions in which the deformation should be zero

        boundary_regularities: how fast the splines go to zero on the boundary
                               for each dimension
                               [0,..,0] : they don't go to zero
                               [1,..,1] : they go to zero with C^0 regularity
                               [2,..,2] : they go to zero with C^1 regularity
        """
        import time
        t1 = time.time()
        self.boundary_regularities = [o - 1 for o in orders] \
            if boundary_regularities is None else boundary_regularities
        # information on B-splines
        self.dim = len(bbox)  # geometric dimension
        self.bbox = bbox
        self.orders = orders
        self.levels = levels
        if isinstance(fixed_dims, int):
            fixed_dims = [fixed_dims]
        self.fixed_dims = fixed_dims
        self.construct_knots()
        self.comm = mesh.mpi_comm()
        # create temporary self.mesh_r and self.V_r to assemble innerproduct
        if self.dim == 2:
            nx = len(self.knots[0]) - 1
            ny = len(self.knots[1]) - 1
            Lx = self.bbox[0][1] - self.bbox[0][0]
            Ly = self.bbox[1][1] - self.bbox[1][0]
            meshloc = fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True,
                                       comm=self.comm)  # quads or triangle?
            # shift in x- and y-direction
            meshloc.coordinates.dat.data[:, 0] += self.bbox[0][0]
            meshloc.coordinates.dat.data[:, 1] += self.bbox[1][0]
            # inner_product.fixed_bids = [1,2,3,4]

        elif self.dim == 3:
            # maybe use extruded meshes, quadrilateral not available
            nx = len(self.knots[0]) - 1
            ny = len(self.knots[1]) - 1
            nz = len(self.knots[2]) - 1
            Lx = self.bbox[0][1] - self.bbox[0][0]
            Ly = self.bbox[1][1] - self.bbox[1][0]
            Lz = self.bbox[2][1] - self.bbox[2][0]
            meshloc = fd.BoxMesh(nx, ny, nz, Lx, Ly, Lz, comm=self.comm)
            # shift in x-, y-, and z-direction
            meshloc.coordinates.dat.data[:, 0] += self.bbox[0][0]
            meshloc.coordinates.dat.data[:, 1] += self.bbox[1][0]
            meshloc.coordinates.dat.data[:, 2] += self.bbox[2][0]
            # inner_product.fixed_bids = [1,2,3,4,5,6]
        t2 = time.time()
        print("meshloc", t2 - t1)
        self.mesh_r = meshloc
        maxdegree = max(self.orders) - 1

        # Bspline control space
        self.V_control = fd.VectorFunctionSpace(self.mesh_r, "CG", maxdegree)
        self.I_control = self.build_interpolation_matrix(self.V_control)
        t3 = time.time()
        print("I_control", t3 - t2)
        # standard construction of ControlSpace
        self.mesh_r = mesh
        element = fd.VectorElement("CG", mesh.ufl_cell(), maxdegree)
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

        assert self.dim == self.mesh_r.geometric_dimension()

        # assemble correct interpolation matrix
        self.FullIFW = self.build_interpolation_matrix(self.V_r)
        t4 = time.time()
        print("FullIFW", t4 - t3)
    def construct_knots(self):
        """
        construct self.knots, self.n, self.N

        self.knots is a list of np.arrays (one per geometric dimension)
        each array corresponds to the knots used to define the spline space

        self.n is a list of univariate spline space dimensions
            (one per geometric dim)

        self.N is the dimension of the scalar tensorized spline space
        """
        self.knots = []
        self.n = []
        for dim in range(self.dim):
            order = self.orders[dim]
            level = self.levels[dim]

            assert order >= 1
            # degree = order-1 # splev uses degree, not order
            assert level >= 1  # with level=1 only bdry Bsplines

            knots_01 = np.concatenate((np.zeros((order - 1,), dtype=float),
                                       np.linspace(0., 1., 2**level + 1),
                                       np.ones((order - 1,), dtype=float)))

            (xmin, xmax) = self.bbox[dim]
            knots = (xmax - xmin) * knots_01 + xmin
            self.knots.append(knots)
            # dimension of univariate spline spaces
            # the "-2" is because we want homogeneous Dir bc
            n = len(knots) - order - 2 * self.boundary_regularities[dim]
            assert n > 0
            self.n.append(n)

        # dimension of multivariate spline space
        N = reduce(lambda x, y: x * y, self.n)
        self.N = N

    def build_interpolation_matrix(self, V):
        """
        Construct the matrix self.FullIFW.

        The columns of self.FullIFW are the interpolant
        of (vectorial tensorized) Bsplines into V
        """
        # construct list of scalar univariate interpolation matrices
        import time
        t1 = time.time()
        interp_1d = self.construct_1d_interpolation_matrices(V)
        t2 = time.time()
        print("    interp_1d", t2 - t1)
        # construct scalar tensorial interpolation matrix
        self.IFWnnz = 0  # to compute sparsity pattern in parallel
        IFW = self.construct_kronecker_matrix(interp_1d)
        t3 = time.time()
        print("    IFW", t3 - t2)
        # interleave self.dim-many IFW matrices among each other
        self.FullIFWnnz = 0  # to compute sparsity pattern in parallel
        M = self.construct_full_interpolation_matrix(IFW)
        t4 = time.time()
        print("    M", t4 - t3)
        return M

    def construct_1d_interpolation_matrices(self, V):
        """
        Create a list of sparse matrices (one per geometric dimension).

        Each matrix has size (M, n[dim]), where M is the dimension of the
        self.V_r.sub(0), and n[dim] is the dimension of the univariate
        spline space associated to the dimth-geometric coordinate.
        The ith column of such a matrix is computed by evaluating the ith
        univariate B-spline on the dimth-geometric coordinate of the dofs
        of self.V_r(0)
        """
        interp_1d = []

        # this code is correct but can be made more beautiful
        # by replacing x_fct with self.id
        x_fct = fd.SpatialCoordinate(self.mesh_r)  # used for x_int
        # compute self.M, x_int will be overwritten below
        x_int = fd.interpolate(x_fct[0], V.sub(0))
        self.M = x_int.vector().size()

        comm = self.comm

        u, v = fd.TrialFunction(V.sub(0)), fd.TestFunction(V.sub(0))
        mass_temp = fd.assemble(u * v * fd.dx)
        self.lg_map_fe = mass_temp.petscmat.getLGMap()[0]

        for dim in range(self.dim):

            order = self.orders[dim]
            knots = self.knots[dim]
            n = self.n[dim]

            # owned part of global problem
            local_n = n // comm.size + int(comm.rank < (n % comm.size))
            I = PETSc.Mat().create(comm=self.comm)
            I.setType(PETSc.Mat.Type.AIJ)
            lsize = x_int.vector().local_size()
            gsize = x_int.vector().size()
            I.setSizes(((lsize, gsize), (local_n, n)))

            I.setUp()
            x_int = fd.interpolate(x_fct[dim], V.sub(0))
            x = x_int.vector().get_local()
            for idx in range(n):
                coeffs = np.zeros(knots.shape, dtype=float)

                # impose boundary regularity
                coeffs[idx + self.boundary_regularities[dim]] = 1
                degree = order - 1  # splev uses degree, not order
                tck = (knots, coeffs, degree)

                values = splev(x, tck, der=0, ext=1)
                rows = np.where(values != 0)[0].astype(np.int32)
                values = values[rows]
                rows_is = PETSc.IS().createGeneral(rows)
                global_rows_is = self.lg_map_fe.applyIS(rows_is)
                rows = global_rows_is.array
                I.setValues(rows, [idx], values)

            I.assemble()  # lazy strategy for kron
            interp_1d.append(I)

        # from IPython import embed; embed()
        return interp_1d

    def vectorkron(self, v, w):
        """
        Compute the kronecker product of two sparse vectors.
        A sparse vector v satisfies: v[idx_] = data_; len_ = len(v)
        This code is an adaptation of scipy.sparse.kron()
        """
        idx1, data1, len1 = v
        idx2, data2, len2 = w

        # lenght of output vector
        lenout = len1*len2

        if len(data1) == 0 or len(data2) == 0:
            # if a vector is zero, the output is the zero vector
            idxout = []
            dataout = []
            return (idxout, dataout, lenout)
        else:
            # rewrite as column vector, multiply, and add row vector
            idxout = (idx1.reshape(len(data1), 1) * len2) + idx2
            dataout = data1.reshape(len(data1), 1) * data2
            return (idxout.reshape(-1), dataout.reshape(-1), lenout)

    def construct_kronecker_matrix(self, interp_1d):
        """
        Construct the tensorized interpolation matrix.

        Do this by computing the kron product of the rows of
        the 1d univariate interpolation matrices.
        In the future, this may be done matrix-free.
        """
        # this is one of the two bottlenecks that slow down initiating Bsplines
        IFW = PETSc.Mat().create(self.comm)
        IFW.setType(PETSc.Mat.Type.AIJ)

        comm = self.comm
        # owned part of global problem
        local_N = self.N // comm.size + int(comm.rank < (self.N % comm.size))
        (lsize, gsize) = interp_1d[0].getSizes()[0]
        IFW.setSizes(((lsize, gsize), (local_N, self.N)))

        # guess sparsity pattern from interp_1d[0]
        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            nnz_ = len(interp_1d[0].getRow(row)[0])  # lenght of nnz-array
            self.IFWnnz = max(self.IFWnnz, nnz_**self.dim)
        IFW.setPreallocationNNZ(self.IFWnnz)
        IFW.setUp()

        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            M = [[A.getRow(row)[0],
                  A.getRow(row)[1], A.getSize()[1]] for A in interp_1d]
            M = reduce(self.vectorkron, M)
            columns, values, lenght = M
            IFW.setValues([row], columns, values)

        IFW.assemble()
        return IFW

    def construct_full_interpolation_matrix(self, IFW):
        """
        Assemble interpolation matrix for vectorial tensorized spline space.
        """
        # this is one of the two bottlenecks that slow down initiating Bsplines
        FullIFW = PETSc.Mat().create(self.comm)
        FullIFW.setType(PETSc.Mat.Type.AIJ)

        # set proper matrix sizes
        d = self.dim
        free_dims = list(set(range(self.dim)) - set(self.fixed_dims))
        dfree = len(free_dims)
        ((lsize, gsize), (lsize_spline, gsize_spline)) = IFW.getSizes()
        FullIFW.setSizes(((d * lsize, d * gsize),
                          (dfree * lsize_spline, dfree * gsize_spline)))

        # (over)estimate sparsity pattern using row with most nonzeros
        # possible memory improvement: allocate precise sparsity pattern
        # row by row (but this needs nnzdiagonal and nnzoffidagonal;
        # not straightforward to do)
        global_rows = self.lg_map_fe.apply([range(lsize)])
        for ii, row in enumerate(range(lsize)):
            row = global_rows[row]
            self.FullIFWnnz = max(self.FullIFWnnz, len(IFW.getRow(row)[1]))
        FullIFW.setPreallocationNNZ(self.FullIFWnnz)

        # preallocate matrix
        FullIFW.setUp()

        # fill matrix by blowing up entries from IFW to do the right thing
        # on vector fields (it's not just a block matrix: values are
        # interleaved as this is how firedrake handles vector fields)
        innerloop_idx = [[i, free_dims[i]] for i in range(dfree)]
        for row in range(lsize):  # for every FE dof
            row = self.lg_map_fe.apply([row])[0]
            # extract value of all tensorize Bsplines at this dof
            (cols, vals) = IFW.getRow(row)
            expandedcols = dfree * cols
            for j, dim in innerloop_idx:
                FullIFW.setValues([d * row + dim],   # global row
                                  expandedcols + j,  # global column
                                  vals)

        FullIFW.assemble()
        return FullIFW

    def restrict(self, residual, out):
        with residual.dat.vec as w:
            self.FullIFW.multTranspose(w, out.vec_wo())

    def interpolate(self, vector, out):
        with out.dat.vec as w:
            self.FullIFW.mult(vector.vec_ro(), w)

    def get_zero_vec(self):
        vec = self.FullIFW.createVecRight()
        return vec

    def get_space_for_inner(self):
        return (self.V_control, self.I_control)

    def visualize_control(self, q, out):
        with out.dat.vec_wo as outp:
            self.I_control.mult(q.vec_wo(), outp)

    def store(self, vec, filename="control.dat"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        viewer = PETSc.Viewer().createBinary(filename, mode="w")
        viewer.view(vec.vec_ro())

    def load(self, vec, filename="control.dat"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        viewer = PETSc.Viewer().createBinary(filename, mode="r")
        vec.vec_wo().load(viewer)


class WaveletControlSpace(BsplineControlSpace):
    def __init__(self, mesh, bbox, orders, dual_orders, level, fixed_dims=[],
                 deriv_orders=[0], boundary_regularities=None):
        import time
        t1 = time.time()
        self.dim = len(bbox)
        self.bbox = bbox
        self.orders = orders
        self.dual_orders = dual_orders
        self.J = level
        if isinstance(deriv_orders, int):
            deriv_orders = [deriv_orders]
        self.deriv_orders = deriv_orders
        self.boundary_regularities = boundary_regularities
        self.precompute()
        t2 = time.time()
        print("precompute", t2 - t1)
        levels = [level] * self.dim
        super().__init__(mesh, bbox, orders, levels, fixed_dims,
                         boundary_regularities)

    def precompute(self):
        self.j0 = -1
        for dim in range(self.dim):
            d = self.orders[dim]
            d_t = self.dual_orders[dim]
            assert (d + d_t) % 2 == 0
            j0 = ceil(log2(d + 2 * d_t - 3) + 1)
            self.j0 = max(self.j0, j0)
        assert self.J > self.j0

        s = np.array(self.deriv_orders)
        self.refine = []
        for dim in range(self.dim):
            refine = []
            (xmin, xmax) = self.bbox[dim]
            d = self.orders[dim]
            d_t = self.dual_orders[dim]
            n_bdry = self.boundary_regularities[dim]

            a = self.compute_primal_refinement_coeffs(d)
            a_t = self.compute_dual_refinement_coeffs(d, d_t)
            ML = self.construct_primal_ML(d)
            ML_t = self.construct_dual_ML(d, d_t, a, a_t, ML)

            n_prev = 0
            for j in range(self.j0, self.J + 1):
                knots_01 = np.concatenate((np.zeros((d - 1,), dtype=float),
                                           np.linspace(0., 1., 2**j + 1),
                                           np.ones((d - 1,), dtype=float)))
                knots = (xmax - xmin) * knots_01 + xmin
                n = len(knots) - d - 2 * n_bdry

                if j == self.j0:
                    M = 2**(j / 2) / np.sum(2**(j * s)) * np.identity(n)
                    refine.append((0, knots, M, n))
                else:
                    M0, M1 = self.construct_refinement_matrix(
                        j - 1, d, d_t, a, a_t, ML, ML_t)
                    if n_bdry > 0:
                        M0 = M0[n_bdry:-n_bdry, n_bdry:-n_bdry]
                        M1 = M1[n_bdry:-n_bdry, :]
                    M0 *= 2**(j / 2) / np.sum(2**((j - 1) * s))
                    M0[np.abs(M0) < 1e-12] = 0.
                    refine.append((0, knots, M0, n_prev))
                    M1 *= 2**(j / 2) / np.sum(2**((j - 1) * s))
                    M1[np.abs(M1) < 1e-12] = 0.
                    refine.append((1, knots, M1, n - n_prev))
                n_prev = n
            self.refine.append(refine)

    def construct_refinement_matrix(self, j, d, d_t, a, a_t, ML, ML_t):
        M0, M1 = self.initial_completion(j, d, a, ML)
        M0_t = self.construct_dual_refinement_matrix(j, d, d_t, a_t, ML_t)
        M1 = M1 - M0 @ M0_t.T @ M1
        return M0, M1

    def compute_support(self, a):
        L = len(a) - 1
        return -floor(L / 2), ceil(L / 2)

    def compute_primal_refinement_coeffs(self, d):
        l1 = -floor(d / 2)
        l2 = ceil(d / 2)

        a = 2**(1 - d) * \
            np.array([binom(d, k - l1) for k in range(l1, l2 + 1)])
        return a

    def construct_primal_ML(self, d):
        knots = np.concatenate((np.zeros(d - 1), np.arange(3 * d - 3)))
        x = np.arange(2 * d - 2)

        B1 = np.empty((2 * d - 2, d - 1))
        B2 = np.empty((2 * d - 2, 2 * d - 2))
        for k in range(d - 1):
            coeffs = np.zeros(3 * d - 4)
            coeffs[k] = 1.
            b = BSpline(knots, coeffs, d - 1)
            B1[:, k] = b(x / 2)
            B2[:, k] = b(x)
        for k in range(d - 1, 2 * d - 2):
            coeffs = np.zeros(3 * d - 4)
            coeffs[k] = 1.
            b = BSpline(knots, coeffs, d - 1)
            B2[:, k] = b(x)
        ML = np.linalg.solve(B2, B1)
        return ML

    def construct_primal_refinement_matrix(self, j, d, a, ML):
        A = np.zeros((2**(j + 1) - d + 1, 2**j - d + 1))
        for k in range(2**j - d + 1):
            A[2*k:2*k+d+1, k] = a

        M0 = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M0[:2*d-2, :d-1] = ML
        M0[d-1:2**(j+1), d-1:2**j] = A
        M0[-(2*d-2):, -(d-1):] = ML[::-1, ::-1]
        return M0 / np.sqrt(2)

    def compute_dual_refinement_coeffs(self, d, d_t):
        l1_t = -floor(d / 2) - d_t + 1
        l2_t = ceil(d / 2) + d_t - 1
        K = (d + d_t) // 2

        a_t = []
        for k in range(l1_t, l2_t + 1):
            a_k = 0
            for n in range(K):
                for i in range(2 * n + 1):
                    a_k += 2**(1 - d_t - 2 * n) * (-1)**(n + i) \
                           * binom(d_t, k + floor(d_t / 2) - i + n) \
                           * binom(K - 1 + n, n) * binom(2 * n, i)
            a_t.append(a_k)
        return np.array(a_t)

    def construct_dual_ML(self, d, d_t, a, a_t, ML):
        l1, l2 = self.compute_support(a)
        l1_t, l2_t = self.compute_support(a_t)

        # make the size of ML compatible with ML_t
        ML_full = np.zeros((2*d + 3*d_t - 5, d + d_t - 2))
        ML_full[:2*d-2, :d-1] = ML
        for k in range(d-1, d+d_t-2):
            ML_full[2*k-d+1:2*k+2, k] = a
        ML_t = np.zeros((2*d + 3*d_t - 5, d + d_t - 2))

        # Compute block of ML_t corresponding to k = d-2, ..., d+2*d_t-3

        # Compute alpha_{0,r}
        alpha0 = np.zeros(d_t)
        alpha0[0] = 1
        for r in range(1, d_t):
            for k in range(l1, l2 + 1):
                sum = 0
                for s in range(r):
                    sum += binom(r, s) * k**(r - s) * alpha0[s]
                alpha0[r] += a[k-l1] * sum
            alpha0[r] /= (2**(r+1) - 2)

        # Compute alpha_{k,r}
        def alpha(k, r):
            res = 0
            for i in range(r + 1):
                res += binom(r, i) * k**i * alpha0[r - i]
            return res

        # Compute beta_{n,r}
        def beta(n, r):
            res = 0
            for k in range(ceil((n - l2_t) / 2), -l1_t):
                res += alpha(k, r) * a_t[n - 2*k - l1_t]
            return res

        def divided_diff(f, t):
            if t.size == 1:
                return f(t[0])
            return (divided_diff(f, t[1:]) - divided_diff(f, t[:-1])) \
                / (t[-1] - t[0])

        D1 = np.zeros((d_t, d_t))
        D2 = np.zeros((d_t, d_t))
        D3 = np.zeros((d_t, d_t))
        k0 = -l1_t - 1
        for n in range(d_t):
            for k in range(n+1):
                D1[n, k] = binom(n, k) * alpha0[n - k]
                D2[n, k] = binom(n, k) * k0**(n - k) * (-1)**k
                D3[n, k] = factorial(k) \
                    * divided_diff(lambda x: x**n, np.arange(k + 1))
        D_t = (D1 @ D2 @ D3)[:, ::-1]
        block1 = np.empty((d + 3*d_t - 3, d_t))
        block1[:d_t, :] = \
            D_t.T @ np.diag([2**(-r) for r in range(d_t)])
        block1[d_t:, :] = np.array([[beta(n - l1_t, r) for r in range(d_t)]
                                    for n in range(d + 2*d_t - 3)])
        ML_t[d-2:, d-2:] = block1 @ np.linalg.inv(D_t.T)

        # Compute block of ML_t corresponding to k = 0, ..., d-3

        def compute_gramian():
            n = ML_full.shape[1]
            UL = ML_full[:n, :]
            LL = ML_full[n:, :]
            UL_t = ML_t[:n, :]
            LL_t = ML_t[n:, :]
            lhs = 2 * np.identity(n**2) - np.kron(UL_t.T, UL.T)
            rhs = (LL.T @ LL_t).reshape(-1, order='F')
            gamma = np.linalg.solve(lhs, rhs)
            return gamma.reshape((n, n), order='F')

        gramian_full = np.identity(2*d + 3*d_t - 5)
        for k in range(d - 3, -1, -1):
            gramian_full[:d+d_t-2, :d+d_t-2] = compute_gramian()
            B_k = ML_full[:, :k+d].T @ gramian_full[:, k+1:2*k+d+1] / 2.

            delta = np.zeros(k+d)
            delta[k] = 1
            ML_t[k+1:2*k+d+1, k] = np.linalg.solve(B_k, delta)

        # Biorthogonalization

        gramian = compute_gramian()
        ML_t[:d+d_t-2, :d+d_t-2] = gramian @ ML_t[:d+d_t-2, :d+d_t-2]
        ML_t = ML_t @ np.linalg.inv(gramian)
        return ML_t

    def construct_dual_refinement_matrix(self, j, d, d_t, a_t, ML_t):
        A_t = np.zeros((2**(j+1) - d - 2*d_t + 3, 2**j - d - 2*d_t + 3))
        for k in range(2**j - d - 2*d_t + 3):
            A_t[2*k:2*k+d+2*d_t-1, k] = a_t

        M0_t = np.zeros((2**(j + 1) + d - 1, 2**j + d - 1))
        M0_t[:2*d+3*d_t-5, :d+d_t-2] = ML_t
        M0_t[d+d_t-2:2**(j+1)-d_t+1, d+d_t-2:2**j-d_t+1] = A_t
        M0_t[-(2*d+3*d_t-5):, -(d+d_t-2):] = ML_t[::-1, ::-1]
        return M0_t / np.sqrt(2)

    def initial_completion(self, j, d, a, ML):
        l1, l2 = self.compute_support(a)
        p = 2**j - d + 1
        q = 2**(j + 1) - d + 1

        P = np.identity(q + 2*d - 2)
        P[:2*d-2, :d-1] = ML / np.sqrt(2)
        P[-(2*d-2):, -(d-1):] = ML[::-1, ::-1] / np.sqrt(2)

        M0 = self.construct_primal_refinement_matrix(j, d, a, ML)
        A = M0[d-1:q+d-1, d-1:p+d-1]
        H_inv = np.identity(q)
        for i in range(d):
            if i % 2 == 0:
                m = i // 2
                repeat = p + min((d-m-1)//2, 1)
                U = np.array([[1, -A[m, 0] / A[m+1, 0]], [0, 1]])
                U = np.kron(np.identity(repeat), U)
                U_inv = np.array([[1, A[m, 0] / A[m+1, 0]], [0, 1]])
                U_inv = np.kron(np.identity(repeat), U_inv)

                H_i = np.identity(q)
                H_i[m:m+2*repeat, m:m+2*repeat] = U
                H_i_inv = np.identity(q)
                H_i_inv[m:m+2*repeat, m:m+2*repeat] = U_inv

            else:
                m = (i-1) // 2
                repeat = p + min((d-m-1)//2, 1)
                L = np.array([[1, 0], [-A[d-m, 0] / A[d-m-1, 0], 1]])
                L = np.kron(np.identity(repeat), L)
                L_inv = np.array([[1, 0], [A[d-m, 0] / A[d-m-1, 0], 1]])
                L_inv = np.kron(np.identity(repeat), L_inv)

                H_i = np.identity(q)
                H_i[q-m-2*repeat:q-m, q-m-2*repeat:q-m] = L
                H_i_inv = np.identity(q)
                H_i_inv[q-m-2*repeat:q-m, q-m-2*repeat:q-m] = L_inv

            A = H_i @ A
            H_inv = H_inv @ H_i_inv

        F_hat = np.zeros((q + 2*d - 2, 2**j))
        F_hat[d-1:l2+d-2, :l2-1] = np.identity(l2-1)
        F_hat[d-1+l2-1:q+d-1+l1-1:2, l2-1:p+l2-1] = np.identity(p)
        F_hat[l1-(d-1):-(d-1), l1:] = np.identity(-l1)

        H_hat_inv = np.identity(q + 2*d - 2)
        H_hat_inv[d-1:-(d-1), d-1:-(d-1)] = H_inv

        M1 = P @ H_hat_inv @ F_hat
        return M0, M1

    def construct_1d_interpolation_matrices(self, V):
        interp_1d = []

        x_fct = fd.SpatialCoordinate(self.mesh_r)
        x_int = fd.interpolate(x_fct[0], V.sub(0))
        self.M = x_int.vector().size()

        comm = self.comm

        u, v = fd.TrialFunction(V.sub(0)), fd.TestFunction(V.sub(0))
        mass_temp = fd.assemble(u * v * fd.dx)
        self.lg_map_fe = mass_temp.petscmat.getLGMap()[0]

        for dim in range(self.dim):
            d = self.orders[dim]
            d_t = self.dual_orders[dim]
            n_bdry = self.boundary_regularities[dim]
            x_int = fd.interpolate(x_fct[dim], V.sub(0))
            x = x_int.vector().get_local()
            lsize = x_int.vector().local_size()
            gsize = x_int.vector().size()

            mats = []
            for e, knots, mat, n in self.refine[dim]:
                local_n = n // comm.size + int(comm.rank < (n % comm.size))
                I = PETSc.Mat().create(comm=self.comm)
                I.setType(PETSc.Mat.Type.AIJ)
                I.setSizes(((lsize, gsize), (local_n, n)))
                if e == 0:
                    I.setPreallocationNNZ(d)
                else:
                    I.setPreallocationNNZ(d + 2 * d_t - 2)
                I.setUp()

                m = mat.shape[0]
                for idx in range(n):
                    coeffs = np.zeros(knots.shape, dtype=float)
                    coeffs[n_bdry:n_bdry+m] = mat[:, idx]
                    degree = d - 1
                    tck = (knots, coeffs, degree)

                    values = splev(x, tck, der=0, ext=1)
                    rows = np.where(values != 0)[0].astype(np.int32)
                    values = values[rows]
                    rows_is = PETSc.IS().createGeneral(rows)
                    global_rows_is = self.lg_map_fe.applyIS(rows_is)
                    rows = global_rows_is.array
                    I.setValues(rows, [idx], values)

                I.assemble()
                mats.append(I)
            interp_1d.append(mats)

        return list(zip(*interp_1d))

    def construct_kronecker_matrix(self, interp_1d):
        IFW = PETSc.Mat().create(self.comm)
        IFW.setType(PETSc.Mat.Type.AIJ)

        comm = self.comm
        local_N = self.N // comm.size + int(comm.rank < (self.N % comm.size))
        (lsize, gsize) = interp_1d[0][0].getSizes()[0]
        IFW.setSizes(((lsize, gsize), (local_N, self.N)))

        d = np.array(self.orders)
        d_t = np.array(self.dual_orders)
        nnz = np.prod(d) + \
            (np.prod(2 * (d + d_t - 1)) - np.prod(d)) * (self.J - self.j0)
        IFW.setPreallocationNNZ(nnz)
        IFW.setUp()

        mats = interp_1d[0]
        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            M = [[A.getRow(row)[0],
                  A.getRow(row)[1], A.getSize()[1]] for A in mats]
            M = reduce(self.vectorkron, M)
            columns, values, length = M
            IFW.setValues([row], columns, values)

        offset = length
        for j in range(self.J - self.j0):
            first = True
            for mats in product(*zip(interp_1d[2*j+1], interp_1d[2*j+2])):
                if first:
                    first = False
                    continue
                for row in range(lsize):
                    row = self.lg_map_fe.apply([row])[0]
                    M = [[A.getRow(row)[0],
                          A.getRow(row)[1], A.getSize()[1]] for A in mats]
                    M = reduce(self.vectorkron, M)
                    columns, values, length = M
                    columns = np.array(columns, dtype=np.int32) + offset
                    IFW.setValues([row], columns, values)
                offset += length

        IFW.assemble()
        return IFW

    def visualize_control(self, q, out=None):
        if out is not None:
            return super().visualize_control(q, out)

        if self.dim == 3:
            msg = "Visualization of 3D control is currently not supported."
            raise NotImplementedError(msg)

        xlim = self.bbox[0]
        ylim = self.bbox[1]
        v = q.vec_ro().array

        def plot(nx, ny, offset):
            n = 2 * nx * ny
            data_x = v[offset:offset+n:2].reshape(nx, ny).T
            data_y = v[offset+1:offset+n:2].reshape(nx, ny).T
            plt.imshow(data_x,
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
            plt.colorbar()
            plt.show()
            plt.imshow(data_y,
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
            plt.colorbar()
            plt.show()
            return offset + n

        offset = 0
        offset = plot(self.refine[0][0][-1], self.refine[1][0][-1], offset)
        for j in range(self.J - self.j0):
            nx0 = self.refine[0][2*j+1][-1]
            nx1 = self.refine[0][2*j+2][-1]
            ny0 = self.refine[1][2*j+1][-1]
            ny1 = self.refine[1][2*j+2][-1]
            offset = plot(nx0, ny1, offset)
            offset = plot(nx1, ny0, offset)
            offset = plot(nx1, ny1, offset)


class ControlVector(ROL.Vector):
    """
    A ControlVector is a variable in the ControlSpace.

    The data of a control vector is a PETSc.vec stored in self.vec.
    If this data corresponds also to a fd.Function, the firedrake wrapper
    around self.vec is stored in self.fun (otherwise, self.fun = None).

    A ControlVector is a ROL.Vector and thus needs the following methods:
    plus, scale, clone, dot, axpy, set.
    """

    def __init__(self, controlspace: ControlSpace, inner_product: InnerProduct,
                 data=None, boundary_extension=None):
        super().__init__()
        self.controlspace = controlspace
        self.inner_product = inner_product
        self.boundary_extension = boundary_extension

        if data is None:
            data = controlspace.get_zero_vec()

        self.data = data
        if isinstance(data, fd.Function):
            self.fun = data
        else:
            self.fun = None

    def from_first_derivative(self, fe_deriv):
        if self.boundary_extension is not None:
            residual_smoothed = fe_deriv.copy(deepcopy=True)
            p1 = fe_deriv
            p1 *= -1
            self.boundary_extension.solve_homogeneous_adjoint(
                p1, residual_smoothed)
            self.boundary_extension.apply_adjoint_action(
                residual_smoothed, residual_smoothed)
            residual_smoothed -= p1
            self.controlspace.restrict(residual_smoothed, self)
        else:
            self.controlspace.restrict(fe_deriv, self)

    def to_coordinatefield(self, out):
        self.controlspace.interpolate(self, out)
        if self.boundary_extension is not None:
            self.boundary_extension.extend(out, out)

    def apply_riesz_map(self):
        """
        Maps this vector into the dual space.
        Overwrites the content.
        """
        # self.controlspace.visualize_control(self)
        self.inner_product.riesz_map(self, self)
        # self.controlspace.visualize_control(self)

    def vec_ro(self):
        if isinstance(self.data, fd.Function):
            with self.data.dat.vec_ro as v:
                return v
        else:
            return self.data

    def vec_wo(self):
        if isinstance(self.data, fd.Function):
            with self.data.dat.vec_wo as v:
                return v
        else:
            return self.data

    def plus(self, v):
        vec = self.vec_wo()
        vec += v.vec_ro()

    def scale(self, alpha):
        vec = self.vec_wo()
        vec *= alpha

    def clone(self):
        """
        Returns a zero vector of the same size of self.

        The name of this method is misleading, but it is dictated by ROL.
        """
        res = ControlVector(self.controlspace, self.inner_product,
                            boundary_extension=self.boundary_extension)
        # res.set(self)
        return res

    def dot(self, v):
        """Inner product between self and v."""
        return self.inner_product.eval(self, v)

    def norm(self):
        return self.dot(self)**0.5

    def axpy(self, alpha, x):
        vec = self.vec_wo()
        vec.axpy(alpha, x.vec_ro())

    def set(self, v):
        vec = self.vec_wo()
        v.vec_ro().copy(vec)

    def __str__(self):
        """String representative, so we can call print(vec)."""
        return self.vec_ro()[:].__str__()
