import firedrake as fd


class InnerProductImpl(object):

    def __init__(self, ls, A):
        self.ls = ls
        self.A = A

    def riesz_map(self, v, out):  # dual to primal
        # expects two firedrake vector objects
        if v.fun is None or out.fun is None:
            self.ls.ksp.solve(v.vec, out.vec)  # Won't do boundary conditions
        self.ls.solve(out.fun, v.fun)

    def eval(self, u, v):  # inner product in primal space
        # expects two firedrake vector objects
        A_u = self.A.createVecLeft()
        self.A.mult(u.vec, A_u)
        return v.vec.dot(A_u)


class InnerProduct(object):

    def __init__(self, fixed_bids=[]):
        self.fixed_bids = fixed_bids
        self.params = self.get_params()

    def get_impl(self, V):
        dim = V.value_size
        if dim == 2:
            zerovector = fd.Constant((0, 0))
        elif dim == 3:
            zerovector = fd.Constant((0, 0, 0))
        else:
            raise NotImplementedError

        a = self.get_weak_form(V)

        nsp = None
        if len(self.fixed_bids) == 0:
            nsp_functions = self.get_nullspace(V)
            if nsp_functions is not None:
                nsp = fd.VectorSpaceBasis(nsp_functions)
                nsp.orthonormalize()

        if len(self.fixed_bids) > 0:
            bc = fd.DirichletBC(V, zerovector, self.fixed_bids)
        else:
            bc = None
        A = fd.assemble(a, mat_type='aij', bcs=bc)

        ls = fd.LinearSolver(A, solver_parameters=self.params, nullspace=nsp,
                             transpose_nullspace=nsp)
        A = fd.as_backend_type(A).mat()
        return InnerProductImpl(ls, A)

    def get_weak_form(self, V):
        raise NotImplementedError

    def get_nullspace(self, V):
        raise NotImplementedError

    def get_params(self):
        return {
                'ksp_solver': 'gmres',
                'pc_type': 'lu',
                'pc_factor_mat_solver_package': 'mumps',
                # 'ksp_monitor': True
                }


class HelmholtzInnerProduct(InnerProduct):

    def get_weak_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx \
            + fd.inner(u, v) * fd.dx
        return a


class LaplaceInnerProduct(InnerProduct):

    def get_weak_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def get_nullspace(self, V):
        dim = V.value_size
        if dim == 2:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0)))
            res = [n1, n2]
        elif dim == 3:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
            n3 = fd.Function(V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
            res = [n1, n2, n3]
        else:
            raise NotImplementedError
        return res


class InterpolatingInnerProduct(InnerProduct):

    def __init__(self, inner_product, interp):
        self.interp = interp
        self.inner_product = inner_product

    def riesz_map(self, v, out):
        # temp = interp*v
        # self.inner_product.riesz_map(temp2, ..)
        # return interpT*...
        pass
