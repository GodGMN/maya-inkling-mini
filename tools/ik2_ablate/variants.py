# variants.py: refined directions - pN: visit N s direction minus its component along the mean answered-harmless state of
# that visit (unit); m21_22: the mean of visits 21 and 22 (unit).  Counts only.
import json,numpy as np
from fit import states  # noqa
hr,ha=states("harmful_train",True),states("harmless_train",False)
u=lambda x:x/np.linalg.norm(x)
for v in (21,22):
    d=u(hr[:,v].mean(0)-ha[:,v].mean(0)); m=u(ha[:,v].mean(0))
    p=u(d-(d@m)*m); print(v,"cos(dir, harmless mean)=%.3f"%(d@m)); p.astype("<f4").tofile(f"dirs/p{v}.f32")
a=np.fromfile("dirs/v21.f32","<f4").astype("f8"); b=np.fromfile("dirs/v22.f32","<f4").astype("f8")
print("cos(v21,v22)=%.3f"%(a@b)); u(a+b).astype("<f4").tofile("dirs/m21_22.f32")
