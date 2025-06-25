#!/usr/bin/env python3
"""
Doosan Robotics - joint_pid_controller_shift.py (v5)
---------------------------------------------------
🔧 Gazebo 명령 미수신 문제 해결:
    ✅ Float64MultiArray 메시지에 layout 명시 추가
    ✅ controller가 요구하는 정확한 topic 및 구조로 publish 보장
"""

import os
import csv
import rclpy
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayLayout


class JointPIDController(Node):
    def __init__(self):
        super().__init__("joint_pid_controller")

        self.num_joints = 6
        self.goals = [
            [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.57, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.57, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.57, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.57],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        self.home_index = len(self.goals) - 1
        self.tol = 0.02
        self.hold_time_required = 0.30

        self.Kp = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
        self.Ki = np.array([0.01, 0.01, 0.01, 0.00, 0.00, 0.00])
        self.Kd = np.array([0.20, 0.15, 0.10, 0.05, 0.05, 0.02])

        self.goal_index = 0
        self.goal = np.array(self.goals[self.goal_index])
        self.integral = np.zeros(self.num_joints)
        self.prev_error = np.zeros(self.num_joints)
        self.prev_time = self.get_clock().now().nanoseconds / 1e9
        self.hold_timer = 0.0
        self.goal_reached = False

        self.pub = self.create_publisher(
            Float64MultiArray,
            "/dsr01/gz/dsr_position_controller/commands",
            10,
        )
        self.sub = self.create_subscription(
            JointState,
            "/dsr01/joint_states",
            self.joint_state_cb,
            10,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"joint_log_{ts}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["time"] + [f"j{j+1}_{t}" for j in range(self.num_joints) for t in ("pos", "goal", "err")]
        self.csv_writer.writerow(header)
        self.time0 = None
        self._t_buf, self._p_buf, self._g_buf, self._e_buf = [], [], [], []

        self.get_logger().info("Controller ready – initial PID gains set. Moving to Goal #1 …")

    def joint_state_cb(self, msg: JointState):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.time0 is None:
            self.time0 = now
        dt = now - self.prev_time
        self.prev_time = now

        current = np.array(msg.position[: self.num_joints])
        error = self.goal - current

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        control = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # === 퍼블리시 메시지에 layout 명시 ===
        msg_out = Float64MultiArray()
        msg_out.data = control.tolist()
        msg_out.layout = MultiArrayLayout(dim=[], data_offset=0)
        self.pub.publish(msg_out)

        if np.all(np.abs(error) < self.tol):
            self.hold_timer += dt
            if not self.goal_reached and self.hold_timer >= self.hold_time_required:
                self.goal_reached = True
                self.get_logger().info(f"[DONE] Goal {self.goal_index + 1} reached")
                self.create_timer(0.1, self._advance_goal, oneshot=True)
        else:
            self.hold_timer = 0.0
            self.goal_reached = False

        t_rel = now - self.time0
        row = [f"{t_rel:.4f}"] + [f"{v:.5f}" for trio in zip(current, self.goal, error) for v in trio]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self._t_buf.append(t_rel)
        self._p_buf.append(current.copy())
        self._g_buf.append(self.goal.copy())
        self._e_buf.append(error.copy())

    def _advance_goal(self):
        self.goal_index = (self.goal_index + 1) % len(self.goals)

        if self.goal_index == self.home_index:
            self.Kp = np.roll(self.Kp, 1)
            self.Ki = np.roll(self.Ki, 1)
            self.Kd = np.roll(self.Kd, 1)
            self.get_logger().info(f"[PID] Gains rolled → Kp {self.Kp.tolist()}")

        self.goal = np.array(self.goals[self.goal_index])
        self.integral.fill(0.0)
        self.prev_error.fill(0.0)
        self.hold_timer = 0.0
        self.goal_reached = False
        self.get_logger().info(
            f"[NEXT] Goal {self.goal_index + 1}/{len(self.goals)} → {self.goal.tolist()}"
        )

    def _save_plot(self):
        n = self.num_joints
        plt.figure(figsize=(12, 8))
        t = self._t_buf
        p = np.stack(self._p_buf)
        g = np.stack(self._g_buf)
        e = np.stack(self._e_buf)
        for j in range(n):
            plt.subplot(3, 2, j + 1)
            plt.plot(t, p[:, j], label="Pos")
            plt.plot(t, g[:, j], "--", label="Goal")
            plt.plot(t, e[:, j], ":", label="Err")
            plt.title(f"Joint {j + 1}")
            plt.xlabel("t [s]")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        png = os.path.splitext(self.csv_path)[0] + ".png"
        plt.savefig(png, dpi=150)
        self.get_logger().info(f"Graph saved → {png}")

    def destroy_node(self):
        self.csv_file.close()
        self._save_plot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JointPIDController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt – shutting down …")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
