import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

@jax.jit
def quat_vectorize(q_row):
    return jnp.transpose(jnp.atleast_2d(q_row))

@jax.jit
def quat_toRow(q_col):
    return jnp.transpose(q_col)[0]

@jax.jit
def quat_mult(q, p):
    qs, ps, qv, pv = q[0], p[0], q[1:], p[1:]
    return jnp.hstack(((qs*ps) - jnp.dot(qv.T, pv), (qs*pv) + (ps*qv) + jnp.cross(qv, pv)))

@jax.jit
def quat_inv_v(q):
    return (q*jnp.array([1., -1., -1., -1.])[:, jnp.newaxis]) / (jnp.linalg.norm(q, axis=0)**2)

@jax.jit
def quat_mult_v(q, p):
    qs, ps, qv, pv = q[0, :], p[0, :], q[1:, :], p[1:, :]
    return jnp.vstack(((qs * ps) - jnp.sum(qv * pv, axis=0), (qs * pv) + (ps * qv) + jnp.cross(qv, pv, axis=0)))

@jax.jit
def quat_exp_v(q):
    qs, qv = q[0, :], q[1:, :]
    return jnp.exp(qs) * jnp.vstack((jnp.cos(jnp.linalg.norm(qv, axis=0)), (qv / jnp.linalg.norm(qv, axis=0)) * jnp.sin(jnp.linalg.norm(qv, axis=0))))

@jax.jit
def quat_log_v(q):
    qs, qv = q[0, :], q[1:, :]
    return jnp.vstack((jnp.log(jnp.linalg.norm(q, axis=0)), (qv/jnp.linalg.norm(qv, axis=0))*jnp.arccos(qs/jnp.linalg.norm(q, axis=0))))

if __name__ == "__main__":
    q = [1, 0, 0.707, 0.707]
    q_vec = quat_vectorize(q)
    print(f'Quaternion:\n{q_vec}')
    q_exp = quat_exp_v(q_vec)
    print(f'Quaternion Exp:\n{q_exp}')
    q_realized = quat_log_v(q_exp)
    print(f'Quaternion Realized:\n{q_realized}')