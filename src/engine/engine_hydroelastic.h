#ifndef MUJOCO_SRC_ENGINE_ENGINE_HYDROELASTIC_H_
#define MUJOCO_SRC_ENGINE_ENGINE_HYDROELASTIC_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

// update aref
void update_aref(const mjModel *m, mjData *d, const int update_qvel);

#endif  // MUJOCO_SRC_ENGINE_ENGINE_HYDROELASTIC_H_
