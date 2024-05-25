import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
Nphi = 256
Ntheta = 128
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2   # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter
timestep = 600 * second
stop_sim_time = 320 * hour
dtype = np.float64


for id in range(2):
    # Bases
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

    # Fields
    u = dist.VectorField(coords, name='u', bases=basis)
    h = dist.Field(name='h', bases=basis)

    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))

    # Initial conditions: zonal jet
    phi, theta = dist.local_grids(basis)
    lat = np.pi / 2 - theta + 0 * phi
    # umax = 80 * meter / second
    lat0 = np.pi / 7
    lat1 = np.pi / 2 - lat0
    en = np.exp(-4 / (lat1 - lat0)**2)

    ##############################################################################################
    # Slightly modify here to create symmetric phenomena on the northern and southern hemisphere.
    ##############################################################################################
    jet1 = (lat0 <= lat) * (lat <= lat1)
    jet2 = (-lat1 <= lat) * (lat <= -lat0)

    print(lat.shape, jet1.shape, jet2.shape, u['g'][0].shape)
    umax = np.random.uniform(low=60.0, high=80.0, size=1) * meter / second
    print(umax / meter * second)
    u_jet1 = umax / en * np.exp(1 / (lat[jet1] - lat0) / (lat[jet1] - lat1))
    u_jet2 = umax / en * np.exp(1 / (lat[jet2] + lat0) / (lat[jet2] + lat1))
    u['g'][0][jet1] = u_jet1
    u['g'][0][jet2] = u_jet2

    # Initial conditions: balanced height
    c = dist.Field(name='c')
    problem = d3.LBVP([h, c], namespace=locals())
    problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
    problem.add_equation("ave(h) = 0")
    solver = problem.build_solver()
    solver.solve()

    # Initial conditions: perturbation
    lat2 = np.pi / 4
    hpert = 120 * meter
    alpha = 1 / 3
    beta = 1 / 15
    ##############################################################################################
    # Slightly modify here to create symmetric phenomena on the northern and southern hemisphere.
    ##############################################################################################
    h['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * (np.exp(-((lat2-lat)/beta)**2) + np.exp(-((lat2+lat)/beta)**2))

    # Problem
    problem = d3.IVP([u, h], namespace=locals())
    problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
    problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

    # Solver
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    # Analysis
    file_name = 'snapshots_test_' + str(int(id))
    snapshots = solver.evaluator.add_file_handler(file_name, sim_dt=1*hour, max_writes=10)
    snapshots.add_task(h, name='height')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    print(id, umax, file_name)

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' % (solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
