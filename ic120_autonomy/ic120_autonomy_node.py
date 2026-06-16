'''
Create Date : 2026/06/12
Author : Ryoya SATO
License : Apach-2.0
'''

import csv
import threading
import math
import os
from dataclasses import dataclass
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from com3_msgs.msg import JointCmd

'''
[データ構造] Waypoiny(ウェイポイント)
CSVファイルから読み込んだ1行分の目標地点データを格納する入れ物です
'''

@dataclass
class Waypoint:
    wp_id : int             # ウェイポイントの番号
    pose  : Pose            # 目標の座標(x, y, z)と姿勢(回転)
    xy_goal_tol : float     # 許容される到着誤差
    des_lin_vel : float     # 目標速度
    state : int             # この地点での役割(0:通過, 1:放土場所, 2:ホーム/積込場所)

'''
[メインクラス] IC120AutonomyNode
クローラダンプ(ic120)の「自律移動」と「荷台(ベゼル)の制御」を
総合的に管理するクラス
'''
class IC120AutonomyNode(Node):
    def __init__(self):
        super().__init__("ic120_autonomy_node")

        self._current_index = 0

        # --- 通信品質(QoS)の設定 ---
        # TRANSIENT_LOCAL : 相手があとから起動しても, 最新のメッセージを確実に届ける設定
        reliable_qos = QoSProfile( reliability=ReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        
        # ========================================================
        # --- パラメータ宣言 ---
        # 外部から変更可能な設定値(ファイル名や判定距離等)を準備
        # ========================================================

        self.declare_parameter( "filename", "waypoints.csv")
        self.declare_parameter( "action_server_name", "navigation_to_pose")
        self.declare_parameter( "global_frame", "map")
        self.declare_parameter( "switch_distance", 2.0) # 次の目的地へ切り替える接近距離[m]
        self.declare_parameter( "odom_topic", "/ic120_0/odom") # 現在地を取得するトピック名

        self._action_server_name = self.get_parameter( "action_server_name").value
        self._global_frame = self.get_parameter( "global_frame").value
        self._switch_distance = self.get_parameter("switch_distance").value
        self._odom_topic = self.get_parameter("odom_topic").value
        waypoints_filename = self.get_parameter( "filename").value

        # クローラの左右モータへの直接指示用パブリッシャと, Nv2からの速度指示を受け取るサブスクライバ
        self.pub_track_cmd = self.create_publisher( JointCmd, '/ic120_0/track_cmd', 10)
        self.sub_cmd_vel = self.create_subscription( Twist, '/cmd_vel', self.cb_bridge_cmd_vel, reliable_qos)

        # ============================================================
        # --- [2] Task Manager(全体の管理)との通信網 ---
        # 指示を受け取る(sub)ための窓口と, 報告を送る(pub)ための窓口
        # ============================================================

        self.sub_tm_transport   = self.create_subscription( Bool, '/ic120/start_transport', self.cb_start_transport, reliable_qos)
        self.sub_tm_tilt_up     = self.create_subscription( Bool, '/ic120/tilt_up_cmd', self.cb_tilt_up, reliable_qos)
        self.sub_tm_tilt_down   = self.create_subscription( Bool, '/ic120/tilt_down_cmd', self.cb_tilt_down, reliable_qos)
        self.sub_tm_return      = self.create_subscription( Bool, '/ic120/start_return', self.cb_start_return, reliable_qos)
        
        self.pub_tm_arrived_dump    = self.create_publisher( Bool, '/ic120/arrived_dump', reliable_qos)
        self.pub_tm_dump_done       = self.create_publisher( Bool, '/ic120/dump_completed', reliable_qos)
        self.pub_tm_tilt_down_done  = self.create_publisher( Bool, '/ic120/tilt_down_completed', reliable_qos)
        self.pub_tm_arrived_home    = self.create_publisher( Bool, '/ic120/arrived_home', reliable_qos)
        self.pub_cmd_vel_passthrough = self.create_publisher( Twist, '/ic120_0/cmd_vel', reliable_qos)

        # ====================================================
        # --- [3] ic120本体との通信網とNav2クライアント ---
        # ====================================================

        qos = QoSProfile( depth = 10, history = QoSHistoryPolicy.KEEP_LAST, durability = QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._start_release_pub = self.create_publisher( Bool, '/start_release', qos) # 荷台上げ開始指示
        self._sub_end_release = self.create_subscription( Bool, '/end_release', self.cb_end_release, 10) # 荷台上げ完了報告

        # Nav2(自動ナビゲーションシステム)へ目的地を送信するためのクライアント
        self._action_client = ActionClient( self, NavigateToPose, self._action_server_name)

        # 現在地(オドメトリ)を常に取得するためのサブスクライバ 
        self._latest_odom = None
        self.sub_odom = self.create_subscription(Odometry, self._odom_topic, self.cb_odom, 10)
        
        # ==================================
        # --- [4]  状態管理フラグの初期化 ---
          ロボットが現在「どんな指示を待っているか」を記録する変数群
        # ==================================
        self.tm_authorized_transport = False
        self.tm_authorized_tilt_up   = False
        self.tm_authorized_tilt_down = False
        self.tm_authorized_return    = False

        self.current_state_index = 0
        self._state = "idle"            # 初期状態は「待機中」
        self._goal_done = False
        self._goal_status = GoalStatus.STATUS_UNKNOWN
        self._goal_lock = threading.Lock()

        # 先行ゴール(止まらずに滑らかに走るため)の判定用フラグ
        self._last_preempt_base_index = -1
        self._started_moving = False
        self._current_goal_pose = None

        # =======================================================
        # --- [5] 初期化処理の実行(CSV読み込みとタイマー起動)
        # =======================================================

        self._waypoints: List[Waypoint] = self._load_waypoints( waypoints_filename)
        if not self._waypoints:
            self.get_logger().error("ウェイポイントが読み込めませんでした。終了します。")
            return
        self.get_logger().info("--- IC120 AUTONOMY READR (Waiting for Task Manager) ---")

        # 0.1秒ごとに_on?timer関数を呼び出すメインループタイマー
        self._timer = self.create_timer( 0.1, self._on_timer)
    
    # =================================================================
    # --- コールバック群 ---
    # トピック(メッセージ)を受信したときに自動で呼ばれる関数群
    # ここでは複雑な処理を行わず, 主に「フラグを立てる」ことだけを行う
    # =================================================================

    def cb_odom(self, msg: Odometry):
        # 現在地を常に最新に更新
        self._latest_odom = msg

    def cb_start_transport( self, msg):
        # Task Managerから搬送開始の指示を受信
        if msg.data and not self.tm_authorized_transport:
            self.get_logger().info("[Task Manager] 指示を受信:搬送を開始")
            self.tm_authorized_transport = True
            $ もし出発時に初期位置(0)にいる場合, 早とちりで終了報告しないよう次(1)を目指す
            if self._current_index == 0 and len(self._waypoints) > 1:
                self._current_index = 1

    def cb_tilt_up( self, msg):
        # Task Managerから荷台(ベゼル)を上げる指示を受信
        if msg.data and not self.tm_authorized_tilt_up:
            self.get_logger().info("[Task Manager] 指示を受信:ベゼルを上げる")
            self.tm_authorized_tilt_up = True

    def cb_tilt_down( self, msg):
        # Task Managerから荷台を下げる指示を受信
        if msg.data and not self.tm_authorized_tilt_down:
            self.get_logger().info("[Task Manager] 指示を受信:ベゼルを下げる")
            self.tm_authorized_tilt_down = True

    def cb_start_return( self, msg):
        # Task Managerから帰還指示を受信
        if msg.data and not self.tm_authorized_return:
            self.get_logger().info("[Task Manager] 指示を受信:定位置へ帰還します")
            self.tm_authorized_return = True

    def cb_end_release( self, msg):
        # ic120本体から荷台が上がりきった報告を受信
        if msg.data and self._state == "waiting_hardware_release":
            self.get_logger().info("[Hardware] ベゼル上昇完了を確認")
            report_msg = Bool()
            report_msg.data = True
            # Task Managerへ放土完了を報告
            self.pub_tm_dump_done.publish( report_msg)
            # 次は荷台を下げる指示を待つ状態へ移行
            self._state = "waiting_tm_tilt_down"

    # --- メインループ ---
    def _on_timer( self):
        if self._state == "finished":
            return

        if self._state == "idle":
            if self.tm_authorized_transport:
                self._send_goal( self._current_index)
            return 

        if self._state == "waiting_for_result":
            with self._goal_lock:
                if self._goal_done:
                    status = self._goal_status
                    self._goal_done = False
                else:
                    status = None
            if status is not None:
                self._handle_goal_result( status)

            # ★ここで「先行ゴール」の判定を常に行う
            self._maybe_preempt_to_next_waypoint()

        if self._state == "waiting_tm_tilt_up":
            if self.tm_authorized_tilt_up:
                self.get_logger().info("指示を受信：ベゼルアップ開始")
                msg = Bool()
                msg.data = True
                self._start_release_pub.publish( msg)
                self._state = "waiting_hardware_release"
                self.tm_authorized_tilt_up = False
        
        if self._state == "waiting_tm_tilt_down":
            if self.tm_authorized_tilt_down:
                report_msg = Bool()
                report_msg.data = True
                self.pub_tm_tilt_down_done.publish( report_msg)
                self._state = "waiting_tm_return"
                self.tm_authorized_tilt_down = False

        if self._state == "waiting_tm_return":
            if self.tm_authorized_return:
                self._go_to_next_waypoint()
                self.tm_authorized_return = False

    # --- Nav2 処理関連 ---
    def _send_goal( self, index):
        wp = self._waypoints[index]
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self._global_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = wp.pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        self._current_index = index
        self._current_goal_pose = wp.pose
        self._started_moving = False
        self._state = "waiting_for_result"
        
        self.get_logger().info(f"Nav2 移動開始: index={index}, state={wp.state}")
        send_future = self._action_client.send_goal_async( goal_msg)
        send_future.add_done_callback( lambda future, g_idx = index: self._goal_response_callback( future, g_idx))

    def _goal_response_callback( self, future, goal_index):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 Goal が拒否されました")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback( lambda future, g_idx = goal_index: self._get_result_callback(future, g_idx))

    def _get_result_callback( self, future, goal_index):
        # 先行ゴールによって古いゴールがキャンセルされた場合は無視する
        if goal_index != self._current_index:
            return
        with self._goal_lock:
            self._goal_done = True
            self._goal_status = future.result().status

    def _maybe_preempt_to_next_waypoint(self):
        if self._latest_odom is None or self._state != "waiting_for_result":
            return

        idx = self._current_index
        wp = self._waypoints[idx]

        # 止まるべきポイント(放土やホーム)では先行ゴールしない
        if wp.state != 0:
            return
        if idx + 1 >= len(self._waypoints):
            return
        if self._last_preempt_base_index == idx:
            return
        if self._current_goal_pose is None:
            return

        ox = self._latest_odom.pose.pose.position.x
        oy = self._latest_odom.pose.pose.position.y
        gx = self._current_goal_pose.position.x
        gy = self._current_goal_pose.position.y
        d = math.hypot(ox - gx, oy - gy)

        # 動いたことを検知
        if (not self._started_moving) and d > self._switch_distance:
            self._started_moving = True

        # スイッチ距離以内に入ったら次のゴールへ上書き
        if self._started_moving and d <= self._switch_distance:
            next_idx = idx + 1
            self.get_logger().info(f"接近検知 (残り {d:.2f}m)。次の waypoint へ先行ゴールします: index={next_idx}")
            self._last_preempt_base_index = idx
            self._send_goal(next_idx)

    def _handle_goal_result( self, status):
        wp = self._waypoints[self._current_index]

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"目標地到着：index={self._current_index}")

            if wp.state == 1:
                self.get_logger().info("放土場所に到着しました。Task_Managerに報告します")
                report_msg = Bool()
                report_msg.data = True
                self.pub_tm_arrived_dump.publish( report_msg)
                self._state = "waiting_tm_tilt_up"

            elif wp.state == 2:
                self.get_logger().info("ホームに帰還しました。Task_Managerに報告します")
                report_msg = Bool()
                report_msg.data = True
                self.pub_tm_arrived_home.publish( report_msg)
                self._current_index = 0
                self.tm_authorized_transport = False
                self._state = "idle"
            else:
                self._go_to_next_waypoint()
        else:
            self.get_logger().warn(f"移動が中断されました(status={status})。次のポイントへ進みます。")
            self._go_to_next_waypoint()

    def _go_to_next_waypoint( self):
        self._current_index += 1
        self._started_moving = False
        self._current_goal_pose = None
        
        if self._current_index >= len(self._waypoints):
            self.get_logger().info("全ウェイポイント完了。待機状態に戻ります。")
            self._current_index = 0
            self.tm_authorized_transport = False
            self._state = "idle"
        else:
            self._send_goal( self._current_index)
            
    # --- CSV読み込み ---
    def _load_waypoints( self, filename: str) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        if not os.path.exists( filename):
            home_dir = os.path.expanduser("~")
            fallback_path = os.path.join(home_dir, "ros2_ws", filename)

            if os.path.exists(fallback_path):
                filename = fallback_path
            else:
                self.get_logger().error(f"ファイルが見つかりません: {filename}")
                return waypoints
        try:
            with open(filename, newline="") as csvfile:
                reader = csv.DictReader( csvfile, delimiter=",")
                for row in reader:
                    wp_id = int( row["id"])
                    pose = Pose()
                    pose.position.x = float( row["pos_x"])
                    pose.position.y = float( row["pos_y"])
                    pose.position.z = float( row["pos_z"])
                    pose.orientation.x = float( row["rot_x"])
                    pose.orientation.y = float( row["rot_y"])
                    pose.orientation.z = float( row["rot_z"])
                    pose.orientation.w = float( row["rot_w"])
                    xy_goal_tol = float( row["xy_goal_tol"])
                    des_lin_vel = float( row["des_lin_vel"])
                    state = int(row["state"])
                    waypoints.append( Waypoint(wp_id, pose, xy_goal_tol, des_lin_vel, state))
        except Exception as e:
            self.get_logger().error(f"CSV 読み込みエラー: {e}")
        return waypoints

    def publish_track_command( self, linear_x, angular_z):
        joint_msg = JointCmd()
        joint_msg.joint_name = ['left_sprocket','right_sprocket']
        left_vel = linear_x - angular_z
        right_vel = linear_x + angular_z
        joint_msg.position = [ float(left_vel), float(right_vel)]
        self.pub_track_cmd.publish( joint_msg)

    def cb_bridge_cmd_vel( self, msg):
        # ログが大量に出るのを防ぐためコメントアウトするかデバッグレベルに変更推奨
        # self.get_logger().info(f"Nav2指令を直接転送：Linear={msg.linear.x}")
        self.pub_cmd_vel_passthrough.publish( msg)

def main(args=None):
    rclpy.init(args=args)
    node = IC120AutonomyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
