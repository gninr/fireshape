import firedrake as fd
import fireshape as fs
import ROL
from L2tracking_PDEconstraint import PoissonSolver
from L2tracking_objective import L2trackingObjective

# Set up problem
mesh = fd.Mesh("mesh.msh")

bbox = [(-4., 0.), (-2., 2.)]
primal_orders = [3, 3]
dual_orders = [3, 3]
nsplines = [8, 8]
max_level = 5
deriv_orders = [0, 1]
Q = fs.WaveletControlSpace(mesh, bbox, primal_orders, dual_orders, nsplines,
                           max_level, deriv_orders, tol=1e-4, eta=0.9)
q = fs.WaveletControlVector(Q)

# Set up PDE constraint
rt = 0.2
ct = (-1.95, 0.)
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
        'Gradient Tolerance': 1e-6,
        'Step Tolerance': 1e-7,
        'Iteration Limit': 15
    }
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
