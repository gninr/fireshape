import firedrake as fd
import fireshape as fs
import ROL
from L2tracking_PDEconstraint import PoissonSolver
from L2tracking_objective import L2trackingObjective

# Setup problem
mesh = fd.Mesh("mesh.msh")

bbox = [(-3., -1.), (-1., 1.)]
nx = [1, 1]
primal_orders = [3, 3]
dual_orders = [3, 3]
levels = [2, 2]
deriv_orders = [0, 1]
Q = fs.WaveletControlSpace(mesh, bbox, nx, primal_orders, dual_orders, levels,
                           deriv_orders, zero_bc_flag=False)
inner = fs.H1InnerProduct(Q)
extension = fs.ElasticityExtension(Q.V_r, fixed_subdomains=[2],
                                   direct_solve=True)
q = fs.ControlVector(Q, inner, boundary_extension=extension)

# Setup PDE constraint
rt = 0.5
ct = (-1.9, 0.)
mesh_m = Q.mesh_m
e = PoissonSolver(mesh_m, rt, ct)

# Save state variable evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")

# Create PDEconstrained objective functional
J_ = L2trackingObjective(e, Q, cb=lambda: out.write(e.solution))
J = fs.ReducedObjective(J_, e)

# ROL parameters
params_dict = {
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Step'
            }
        }
    },
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 10
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-5,
        'Iteration Limit': 15
    }
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
