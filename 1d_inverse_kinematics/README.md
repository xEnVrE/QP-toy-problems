# 1-R Robot Inverse Kinematics

## Requirements
- [qpsolvers](https://github.com/stephane-caron/qpsolvers)
- `matplotlib`
- `numpy`

## Theory

Standard inverse kinematics for a 1-R robot solved via QP with joint and joint velocity limits.

## Run as

```console
python 1d_inverse_kinematics/oned_ik.py
```

The expected outcome is:

<img src="https://github.com/xEnVrE/QP-toy-problems/blob/master/1d_inverse_kinematics/assets/example.png" width=1000></img>

## References

[Inverse kinematics](https://scaron.info/robot-locomotion/inverse-kinematics.html)

[Conversion from least squares to quadratic programming](https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html)

[Real-time prioritized kinematic control under inequality constraints for redundant manipulators (RSS VII)](http://www.roboticsproceedings.org/rss07/p21.pdf)
