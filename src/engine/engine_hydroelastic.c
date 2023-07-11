#include "engine/engine_hydroelastic.h"

#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_io.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_sparse.h"

static double RegularizedFriction(const double s, const double mu)
{
  if (s >= 1)
  {
    return mu;
  }
  else
  {
    return mu * s * (2.0 - s);
  }
}

int CrossesTheStictionRegion(const mjModel *m, mjData *d,
                             const mjtNum *v,
                             const mjtNum *dv,
                             const double v_dot_dv, const double dv_norm, const double dv_norm2,
                             double epsilon_v, double v_stiction, double *alpha_out)
{
  double alpha = *alpha_out;
  if (v_dot_dv < 0.0)
  {                               // Moving towards the origin.
    alpha = -v_dot_dv / dv_norm2; // alpha > 0
    if (alpha < 1.0)
    { // The update might be crossing the stiction region.
      mjtNum *v_alpha = mj_stackAlloc(d, 2);
      mju_addScl(v_alpha, v, dv, alpha, 2); // Note: v_alpha.dot(dv) = 0.
      const double v_alpha_norm = mju_norm(v_alpha, 2);
      if (v_alpha_norm < epsilon_v)
      {
        // v_alpha is almost zero.
        // This situation happens when dv ≈ -a v with a > 0. Therefore we cap
        // v_alpha to v_alpha = v / ‖v‖⋅vₛ/2. To do this, we "move" v_alpha
        // in the direction opposite of dv/‖dv‖ by magnitude vs/2, or similarly,
        // we subtract ‖dv‖vs/2 from the previously computed alpha.
        alpha -= v_stiction / 2.0 / dv_norm;
        // DRAKE_ASSERT(0 < alpha && alpha <= 1);
        *alpha_out = alpha;
        return 1; // Crosses the stiction region.
      }
      else if (v_alpha_norm < v_stiction)
      {
        // v_alpha falls within the stiction region but its magnitude is
        // larger than epsilon_v.
        *alpha_out = alpha;
        return 1; // Crosses the stiction region.
      }
    }
  }
  return 0;
}

double SolveQuadraticForTheSmallestPositiveRoot(
    const double a, const double b, const double c)
{
  // using std::abs;
  // using std::max;
  // using std::min;
  // using std::sqrt;
  double alpha;
  // First determine if a = 0 (to machine epsilon). This comparison is fair
  // since a is dimensionless.
  if (mju_abs(a) < __DBL_EPSILON__)
  {
    // There is only a single root to the, now linear, equation bα + c = 0.
    alpha = -c / b;
    // Note: a = 0, α > 0 => x_dot_dx = x * dx * cmin ≠ 0 => b ≠ 0
    // We assert this even though within the scope this method is called it is
    // not possible for b to be zero.
    // DRAKE_ASSERT(abs(b) > __DBL_EPSILON__);
  }
  else
  {
    // The determinant, Δ = b² - 4ac, of the quadratic equation.
    const double Delta = b * b - 4 * a * c; // Uppercase, as in Δ.
    // Geometry tell us that a real solution does exist i.e. Delta > 0.
    // DRAKE_DEMAND(Delta > 0);
    const double sqrt_Delta = mju_sqrt(Delta);

    // To avoid loss of significance, when 4ac is relatively small compared
    // to b² (i.e. the square root of the discriminant is close to b), we use
    // Vieta's formula (α₁α₂ = c / a) to compute the second root given we
    // computed the first root without precision lost. This guarantees the
    // numerical stability of the method.
    const double numerator = -0.5 * (b + (b > 0.0 ? sqrt_Delta : -sqrt_Delta));
    const double alpha1 = numerator / a;
    const double alpha2 = c / numerator;

    // The geometry of the problem tells us that at least one must be
    // positive.
    // DRAKE_DEMAND(alpha2 > 0 || alpha1 > 0);

    if (alpha2 > 0 && alpha1 > 0)
    {
      // This branch is triggered for large angle changes (typically close
      // to 180 degrees) between v1 and vt.
      alpha = mju_min(alpha2, alpha1);
    }
    else
    {
      // This branch is triggered for small angles changes (typically
      // smaller than 90 degrees) between v1 and vt.
      alpha = mju_max(alpha2, alpha1);
    }
  }
  return alpha;
}

double CalcAlpha_(const mjModel *m, mjData *d, const mjtNum *v, const mjtNum *dv,
                  double cos_theta_max, double v_stiction, double relative_tolerance)
{

  // // εᵥ is used to determine when a velocity is close to zero.
  const double epsilon_v = v_stiction * relative_tolerance;
  const double epsilon_v2 = epsilon_v * epsilon_v;

  mjtNum *v1 = mj_stackAlloc(d, 2);
  mju_add(v1, v, dv, 2); // v_alpha = v + dv, when alpha = 1.
  double dv_squared_norm = mju_norm(dv, 2);
  dv_squared_norm = dv_squared_norm * dv_squared_norm;

  const double v_norm = mju_norm(v, 2);
  const double v1_norm = mju_norm(v1, 2);

  // Case I: Quick exit for small changes in v.
  const double dv_norm2 = dv_squared_norm;
  if (dv_norm2 < epsilon_v2)
  {
    return 1.0;
  }

  // const double v_norm = v_norm;
  // const double v1_norm = v1_norm;
  const double x = v_norm / v_stiction; // Dimensionless slip v and v1.
  const double x1 = v1_norm / v_stiction;
  // From Case I, we know dv_norm > epsilon_v.
  const double dv_norm = sqrt(dv_norm2);

  // Case II: limit transition from stiction to sliding when x << 1.0 and
  // gradients might be close to zero (due to the "soft norms").
  if (x < relative_tolerance && x1 > 1.0)
  {
    // we know v1 != 0  since x1 > 1.0.
    // With v_alpha = v + alpha * dv, we make |v_alpha| = v_stiction / 2.
    // For this case dv ≈ v1 (v ≈ 0). Therefore:
    // alpha ≈ v_stiction / dv_norm / 2.
    return v_stiction / dv_norm / 2.0;
  }

  // Case III: Transition to an almost exact stiction from sliding.
  // We want to avoid v1 landing in a region of zero gradients so we force
  // it to land within the circle of radius v_stiction, at v_stiction/2 in the
  // direction of v.
  if (x > 1.0 && x1 < relative_tolerance)
  {
    // In this case x1 is negligible compared to x. That is dv ≈ -v. For this
    // case we'll limit v + αdv = vₛ/2⋅v/‖v‖. Using v ≈ -dv, we arrive to
    // dv(α-1) = -vₛ/2⋅dv/‖dv‖ or:
    return 1.0 - v_stiction / 2.0 / dv_norm;
  }

  if (x < 1.0)
  {
    // Another quick exit. Two possibilities (both of which yield the same
    // action):
    // x1 < 1: we go from within the stiction region back into it. Since this
    //         region has strong gradients, we allow it. i.e. alpha = 1.0
    // x1 > 1: If we go from a region of strong gradients (x < 1) to sliding
    //         (x1 > 1), we allow it. Notice that the case from weak gradients
    //         (close to zero) when x < relative_tolerance, was covered by
    //         Case II.
    return 1.0;
  }
  else
  { // x > 1.0
    if (x1 < 1.0)
    {
      // Case IV:
      // From Case III we know that x1 > relative_tolerance, i.e x1 falls in a
      // region of strong gradients and thus we allow it.
      return 1.0;
    }

    // Case V:
    // Since v_stiction is so small, the next case very seldom happens. However,
    // it is a very common case for 1D-like problems for which tangential
    // velocities change in sign and the zero crossing might be missed.
    // Notice that since we reached this point, we know that:
    //  - x > 1.0 (we are within the scope of an if statement for x > 1)
    //  - x1 > 1.0 (we went through Case IV)
    //  - dv_norm > epsilon_v (we went through Case I, i.e. non-zero)
    // Here we are checking for the case when the line connecting v and v1
    // intersects the boundary of the stiction region. For this case we
    // compute alpha so that the update corresponds to the velocity closest
    // to the origin.
    double alpha;
    const double v_dot_dv = mju_dot(v, dv, 2);

    if (CrossesTheStictionRegion(m, d,
                                 v, dv, v_dot_dv, dv_norm, dv_norm2, epsilon_v, v_stiction, &alpha))
    {
      return alpha;
    }

    // If we are here we know:
    //  - x > 1.0
    //  - x1 > 1.0
    //  - dv_norm > epsilon_v
    //  - line connecting v with v1 never goes through the stiction region.
    //
    // Case VI:
    // Therefore we know changes happen entirely outside the circle of radius
    // v_stiction. To avoid large jumps in the direction of v (typically during
    // strong impacts), we limit the maximum angle change between v to v1.
    // To this end, we find a scalar 0 < alpha < 1 such that
    // cos(θₘₐₓ) = v⋅(v+αdv)/(‖v‖‖v+αdv‖), see [Uchida et al., 2015].

    // First we compute the angle change when alpha = 1, between v1 and v.
    // const double cos1 = v.dot(v1) / v_norm / v1_norm;
    const double cos1 = mju_dot(v, v1, 2) / v_norm / v1_norm;

    // We allow angle changes theta < theta_max, and we take alpha = 1.0.
    // In particular, when v1 is exactly aligned with v (but we know it does not
    // cross through zero, i.e. cos(theta) > 0).
    if (cos1 > cos_theta_max)
    {
      return 1.0;
    }
    else
    {
      // we limit the angle change to theta_max so that:
      // cos(θₘₐₓ) = v⋅(v+αdv)/(‖v‖‖v+αdv‖)
      // if we square both sides of this equation, we arrive at a quadratic
      // equation with coefficients a, b, c, for α. The math below simply is the
      // algebra to compute coefficients a, b, c and solve the quadratic
      // equation.

      // All terms are made non-dimensional using v_stiction as the reference
      // scale.
      const double x_dot_dx = v_dot_dv / (v_stiction * v_stiction);
      const double dx = dv_norm / v_stiction;
      const double x2 = x * x;
      const double dx4 = x2 * x2;
      const double cos_theta_max2 = cos_theta_max * cos_theta_max;

      // Form the terms of the quadratic equation aα² + bα + c = 0.
      const double a = x2 * dx * dx * cos_theta_max2 - x_dot_dx * x_dot_dx;
      const double b = 2 * x2 * x_dot_dx * (cos_theta_max2 - 1.0);
      const double c = dx4 * (cos_theta_max2 - 1.0);

      // Solve quadratic equation. We know, from the geometry of the problem,
      // that the roots to this problem are real. Thus, the discriminant of the
      // quadratic equation (Δ = b² - 4ac) must be positive.
      // We use a very specialized quadratic solver for this case where we know
      // there must exist a positive (i.e. real) root.
      alpha = SolveQuadraticForTheSmallestPositiveRoot(a, b, c);

      // The geometry of the problem tells us that α ≤ 1.0
      return alpha;
    }
  }
}

double CalcAlpha(const mjModel *m, mjData *d, const mjtNum *efc_v, const mjtNum *efc_dv)
{
  double alpha = 1.0;
  const double v_stiction = 1.0e-4;

  for (int ic = 0; ic < d->ncon; ++ic)
  {
    mjContact c = d->contact[ic];
    if (c.dim > 1 && !c.exclude && c.hydroelastic_contact)
    {
      alpha = mju_min(alpha, CalcAlpha_(m, d, efc_v + c.efc_address + 1, efc_dv + c.efc_address + 1,
                                        cos(M_PI / 3.0), v_stiction, 1.0e-2));
    }
    ic += c.dim - 1;
  }
  // for (int ic = 0; ic < d->ncon; ++ic) {  // Index ic scans contact points.
  //   const int ik = 2 * ic;  // Index ik scans contact vector quantities.
  //   auto vt_ic = vt.template segment<2>(ik);
  //   const auto dvt_ic = Delta_vt.template segment<2>(ik);
  //   alpha = min(
  //       alpha,
  //       drake::multibody::internal::TalsLimiter<double>::CalcAlpha(
  //           vt_ic, dvt_ic,
  //           cos_theta_max_, v_stiction, parameters_.relative_tolerance));
  // }

  return alpha;
}

double CalcAtanXOverXFromXSquared(const double x2) {
  // We are protecting the computation near x = 0 specifically so that
  // numerical values (say double and AutoDiffXd) do not lead to ill-formed
  // expressions with divisions by zero.
  const double x_cuttoff = 0.12;
  const double x_cutoff_squared = x_cuttoff * x_cuttoff;
  if (x2 <= x_cutoff_squared) {
    // We use the Taylor expansion of f(x)=atan(x)/x below a given cutoff
    // x_cutoff, since neither atan(x)/x nor its automatic derivatives with
    // AutodiffXd can be evaluated at x = 0. However, f(x) is well defined
    // mathematically given its limits from left and right exist. Choosing
    // the value of x_cutoff and the number of terms is done to minimize the
    // amount of round-off errors. We estimated these values by comparing
    // against reference values computed with Variable Precision Arithmetic.
    // For further details please refer to Drake issue #15029 documenting this
    // process.

    // clang-format off
      return 1. -
             x2 * (1. / 3. -
             x2 * (1. / 5. -
             x2 * (1. / 7. -
             x2 * (1. / 9. -
             x2 * (1. / 11. -
             x2 * (1. / 13. -
             x2 * (1. / 15. -
             x2 / 17.)))))));
    // clang-format on
  }
  // using std::atan;
  // using std::sqrt;
  const double x = sqrt(x2);
  return atan(x) / x;
}

void update_aref(const mjModel *m, mjData *d, const int update_qvel)
{
  mjMARKSTACK;
  const double v_stiction = 1.0e-4;
  // const double  vslip_regularizer_ = 1e-6;
  double relative_tolerance = 1.0e-2;
  mjtNum *qvel = mj_stackAlloc(d, m->nv);
  mjtNum *qvel_diff = mj_stackAlloc(d, m->nv);
  mjtNum *efc_qvel_current = mj_stackAlloc(d, d->nefc);
  mjtNum *efc_qvel_diff = mj_stackAlloc(d, d->nefc);
  mju_copy(qvel, d->qvel, m->nv);
  // mju_printMat(qvel, 1, m->nv);

  if (update_qvel)
  {
    // mjtNum *acc = mj_stackAlloc(d, d->nefc);

    // mjtNum *efc_vel = new mjtNum[d->nefc];

    // mju_add(acc, d->qacc, d->qacc_smooth, m->nv);
    // mju_copy(acc, d->qacc, m->nv);
    // mju_addToScl(qvel, acc, m->opt.timestep, m->nv);
    mju_addScl(qvel, d->qacc, d->qacc_smooth, -2, m->nv);
    mju_scl(qvel, qvel, m->opt.timestep, m->nv);
    mju_addTo(qvel, d->qvel_old, m->nv);
    mju_sub(qvel_diff, qvel, d->qvel_current, m->nv);
    mj_mulJacVec(m, d, efc_qvel_current, d->qvel_current);
    mj_mulJacVec(m, d, efc_qvel_diff, qvel_diff);
    // double alpha = 0.2;
    // if (0) {
    double alpha = CalcAlpha(m, d, efc_qvel_current, efc_qvel_diff);
    // }
    mju_addToScl(d->qvel_current, qvel_diff, alpha, m->nv);
    mju_copy(qvel, d->qvel_current, m->nv);

    // mju_scl(d->qacc, d->qacc_smooth, 2, m->nv);
    // mju_addToScl(d->qacc, qvel, 1. / m->opt.timestep, m->nv);
    // mju_addToScl(d->qacc, d->qvel_old, -1. / m->opt.timestep, m->nv);
  }
  else
  {
    mju_copy(d->qvel_old, d->qvel, m->nv);
    mju_copy(d->qvel_current, d->qvel, m->nv);
  }
  mj_mulJacVec(m, d, d->efc_vel, qvel);

  mjtNum *efc_acc_smooth = mj_stackAlloc(d, d->nefc);
  mj_mulJacVec(m, d, efc_acc_smooth, d->qacc_smooth);

  // mju_printMat(qvel, 1, m->nv);


  for (int i = 0; i < d->ncon; ++i)
  {
    // int i0 = idx[i];
    mjContact c = d->contact[i];

    if (!c.hydroelastic_contact)
    {
      continue;
    }

    double damping = c.d;
    
    double stiffness = c.k;
    // printf("%4.2f\n",1.212);
    double fn0 = c.fn0;

    int ic = c.efc_address;

    const double signed_damping_factor = (1.0 -  damping * d->efc_vel[ic]);

    const double damping_factor = mjMAX(0.0, signed_damping_factor);

  
    // fₙ = (1 − d vₙ)₊ (fₙ₀ − h k vₙ)₊

    const double signed_undamped_fn = fn0 - 0 * m->opt.timestep * stiffness * d->efc_vel[ic];
    // printf("fn0 vs  sufn %4.2f %4.2f\n", fn0, signed_undamped_fn);
    const double undamped_fn = mjMAX(0.0, signed_undamped_fn);
    // std::cout << "signed_damping_factor: " << signed_damping_factor << " signed_undamped_fn " << signed_undamped_fn << std::endl;

    // std::cout << d->efc_R[ic] << " " << 1 / d->efc_D[ic]  << std::endl;
    // if (!c.exclude) {
    // mjtNum imp = d->efc_KBIP[4*ic +2];
    // TODO add impedance
    // d->efc_aref[ic] =  (-damping_factor * undamped_fn *    imp / (1.0 - imp)   ) * d->efc_R[ic] ;// + a_smooth[ic];
    // f_drake[ic] =  (damping_factor * undamped_fn   ); // * d->efc_R[ic];// / 2.0 ;// + a_smooth[ic];
    const double fn = damping_factor * undamped_fn;
    // d->efc_aref[ic] = 0* ( - stiffness[i] * fn0[i]* d->efc_R[ic] - d->efc_vel[ic] * damping[i]) + 1 *(damping_factor * undamped_fn   ) * d->efc_R[ic];
    d->efc_aref[ic] = fn * d->efc_R[ic]; // +  efc_acc_smooth[ic];// * d->efc_KBIP[4*ic+2] / (1-d->efc_KBIP[4*ic+2]);// + efc_acc_smooth[ic];

    if (c.dim == 3)
    {

      if (1) {
      const double mu = c.friction[1];

      // The stiction tolerance.

      const double epsilon_v = v_stiction * relative_tolerance;
      const double epsilon_v2 = epsilon_v * epsilon_v;

      mjtNum *vt_ic = d->efc_vel + ic + 1;

      // "soft norm":
      // const double v_slip = (pow(vt_ic[0], 2) + pow(vt_ic[1], 2) + epsilon_v2);
      const double v_slip = sqrt(pow(vt_ic[0], 2) + pow(vt_ic[1], 2) + epsilon_v2);
      // // "soft" tangent vector:
      // // const Vector2<double> that_ic = vt_ic / v_slip;
      // // const Vector2<double> t_hat = that_ic;
      double mu_regularized = RegularizedFriction(v_slip / v_stiction, mu);

      // const double vs_squared = vslip_regularizer_ * vslip_regularizer_ + 0 * mu_regularized;
      // const double x_squared = v_slip / vs_squared;

      // const double regularized_friction = (2.0 / M_PI) * mu * fn *
      //                            CalcAtanXOverXFromXSquared(x_squared) /
      //                            vslip_regularizer_;  // [Ns/m].
      // const Vector3<T> ft_Aq_W = -regularized_friction * vt_BqAq_W;
      // Friction force.
      // Vector2<double> ft = -mu_regularized * that_ic * fn;
      for (int j = 0; j < 2; ++j)
      {
        double that_ic = vt_ic[j] / v_slip;
        double ft = -mu_regularized * that_ic * fn;
        // double ft = -regularized_friction * vt_ic[j];
        int ii = ic + j + 1;
        d->efc_aref[ii] = ft * d->efc_R[ii]; // + efc_acc_smooth[ii];// * d->efc_KBIP[4*ii+2] / (1-d->efc_KBIP[4*ii+2]);// + efc_acc_smooth[ic + j + 1];
      }

      } else {
         for (int j = 0; j < 2; ++j)
      {
        int ii = ic + j + 1;
        d->efc_aref[ii] = mjMAX(0,  fn0 * (0.0 - damping * d->efc_vel[ii])) * d->efc_R[ii];

      }
      }
      
    }
  }
 
  // mju_printMat(d->efc_aref, 1, d->nefc);
  mjFREESTACK;
}
