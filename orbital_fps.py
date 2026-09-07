from pyrr import Vector3, vector3, matrix44
from math import sin, cos, radians
import math
import time
import glfw

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

class CenterOrbitalCamera:
    def __init__(self,
                 center=Vector3([0.0, 0.0, 0.0]),
                 radius=10.0, 
                 globe_radius=5.0, # ADDED: The custom unit radius of your 3D sphere
                 min_radius=5.2, max_radius=25.0,
                 yaw=-90.0,
                 pitch=0.0,
                 mouse_sensitivity=0.1,
                 zoom_sensitivity=4.0,
                 zoom_smoothness=0.1):
        self.center            = Vector3(center)
        self.radius            = radius
        self.globe_radius      = globe_radius # We need this for the math bridge
        self.target_radius     = radius
        self.min_radius        = min_radius
        self.max_radius        = max_radius
        self.last_move_time = time.time()
        self.is_stationary = False # This becomes True ONLY after 0.8s of silence
        self.yaw               = yaw % 360.0
        self.pitch             = clamp(pitch, -89.0, 89.0)

        self.last_radius = radius
        self.last_yaw    = self.yaw
        self.last_pitch  = self.pitch
        self.zoom_vel    = 0.0
        self.yaw_vel     = 0.0
        self.pitch_vel   = 0.0
        self.last_time   = time.time()

        self.mouse_sensitivity = mouse_sensitivity
        self.zoom_sensitivity  = zoom_sensitivity
        self.zoom_smoothness   = zoom_smoothness

        self.position = Vector3([0.0, 0.0, radius])
        self.up       = Vector3([0.0, 1.0, 0.0])
        self._update_vectors()
        self.front = Vector3([0.0, 0.0, -1.0])
        
    def get_view_matrix(self):
        return matrix44.create_look_at(self.position, self.center, self.up)

    def get_position_xyz(self):
        return self.position.x, self.position.y, self.position.z

    def process_mouse_movement(self, x_offset, y_offset, constrain_pitch=True):
        self.yaw   += x_offset * self.mouse_sensitivity
        self.pitch += y_offset * self.mouse_sensitivity
        if constrain_pitch:
            self.pitch = clamp(self.pitch, -89.0, 89.0)
        self._update_vectors()

    def process_scroll(self, y_offset):
        t = (self.radius - self.min_radius) / (self.max_radius - self.min_radius)
        scale = (t ** 2) + 0.1 
        delta = y_offset * self.zoom_sensitivity * scale

        self.target_radius = clamp(
            self.target_radius - delta,
            self.min_radius,
            self.max_radius
        )

    def update(self):
        curr_time = time.time()
        dt = curr_time - self.last_time
        if dt <= 0: dt = 1/60.0

        # Smooth Zoom
        self.radius += (self.target_radius - self.radius) * self.zoom_smoothness

        # Velocity Math
        self.zoom_vel  = (self.last_radius - self.radius) / dt
        dyaw = self.yaw - self.last_yaw
        if dyaw > 180: dyaw -= 360
        elif dyaw < -180: dyaw += 360
        self.yaw_vel   = dyaw / dt
        self.pitch_vel = (self.pitch - self.last_pitch) / dt

        # --- DWELL TIMER LOGIC ---
        # Calculate combined movement speed
        current_speed = abs(self.yaw_vel) + abs(self.pitch_vel) + abs(self.zoom_vel)
        
        # If moving faster than a tiny jitter threshold, reset the clock
        if current_speed > 0.7: 
            self.last_move_time = curr_time
            self.is_stationary = False
        else:
            # If we've been still for more than 0.8 seconds
            if curr_time - self.last_move_time > 0.2:
                if not self.is_stationary:
                    print("CAMERA SETTLED: Starting requests...")
                self.is_stationary = True
        # -------------------------

        self.last_radius = self.radius
        self.last_yaw    = self.yaw
        self.last_pitch  = self.pitch
        self.last_time   = curr_time

        self._update_vectors()

    def _update_vectors(self):
        phi   = radians(self.pitch)
        theta = radians(self.yaw)
        x = self.radius * cos(phi) * cos(theta)
        y = self.radius * sin(phi)
        z = self.radius * cos(phi) * sin(theta)
        
        self.position = self.center + Vector3([x, y, z])
        
        # We need to save 'front' to the class so the raycaster can see it
        self.front = vector3.normalize(self.center - self.position)
        
        right = vector3.normalize(vector3.cross(self.front, Vector3([0.0, 1.0, 0.0])))
        self.up = vector3.normalize(vector3.cross(right, self.front))

    def _get_zoom_level(self, r):
        if r <= self.min_radius: return 15.0
        if r >= self.max_radius: return 8.0
        log_p = math.log2(self.max_radius / r) / math.log2(self.max_radius / self.min_radius)
        return 8.0 + (clamp(log_p, 0.0, 1.0) * 7.0)

    def get_raycast_center(self):
        # 1. Unpack camera position (works for Vector3 or numpy)
        ox, oy, oz = self.position[0], self.position[1], self.position[2]
        
        # 2. Unpack direction vector (standard numpy unpacking)
        # This fixes the AttributeError
        dx, dy, dz = self.front[0], self.front[1], self.front[2]
        
        R = self.globe_radius
        
        # 3. Ray-Sphere Intersection Math (Quadratic)
        b = 2.0 * (ox*dx + oy*dy + oz*dz)
        c = (ox*ox + oy*oy + oz*oz) - (R * R)
        
        discriminant = (b * b) - (4 * c)
        
        if discriminant < 0:
            return None, None 
            
        t = (-b - math.sqrt(discriminant)) / 2.0
        
        # Hit point
        hx = ox + (t * dx)
        hy = oy + (t * dy)
        hz = oz + (t * dz)
        
        # 4. Conversion to Lat/Lon
        center_lat = math.degrees(math.asin(hy / R))
        center_lon = math.degrees(math.atan2(hx, hz))
        #center_lon -= 97.0
        #center_lon -= 88.25
        #center_lon -= 86.75
        center_lon -= 90.0

        # Ensure it stays within -180 to 180
        if center_lon < -180: center_lon += 360
        if center_lon > 180:  center_lon -= 360
        return center_lat, center_lon

    def get_camera_state(self, window):
        # 1. CORE METRICS
        zoom_float = self._get_zoom_level(self.radius)
        
        # --- NEW RAYCAST LOGIC ---
        center_lat, center_lon = self.get_raycast_center()
        
        # Safety catch: If user is looking at space, abort tile generation
        if center_lat is None:
            return None 
        # -------------------------

        # 2. ZOOM THRESHOLD (The Hard Binary Switch)
        is_tactical_zone = zoom_float >= 8.0

        # 3. SPEED CHECK (For pulling data, NOT for generating tiles)
        cam_speed = abs(self.yaw_vel) + abs(self.pitch_vel) + abs(self.zoom_vel)

        # 4. VIEWPORT SPAN CALCULATION
        distance_from_surface = max(self.radius - self.globe_radius, 0.01)
        fov_y_rad = math.radians(45.0)
        visible_y_units = 2.0 * distance_from_surface * math.tan(fov_y_rad / 2.0)
        
        circumference = 2.0 * math.pi * self.globe_radius
        degrees_per_unit = 360.0 / circumference

        # Handle window minimizing/resizing safely
        WIDTH, HEIGHT = glfw.get_window_size(window)
        aspect = WIDTH / HEIGHT if HEIGHT > 0 else 1.0

        visible_lat_span = visible_y_units * degrees_per_unit
        visible_lon_span = visible_lat_span * aspect

        # 5. PADDING & SAFETY CLAMPING
        # Add padding to cover edges, but CLAMP it to a maximum degree span
        # This prevents the while-loop below from freezing the app if you zoom out fast
        pad_lat = min(visible_lat_span * 1.15, 6.0) 
        pad_lon = min(visible_lon_span * 1.15, 8.0) 

        # 6. DYNAMIC GRID GENERATION (Always run this if in tactical zone!)
        tile_requests = []
        
        if is_tactical_zone:
            grid_res = 1.0 
            
            # Snap the edges to the nearest grid line
            min_lon_grid = math.floor((center_lon - pad_lon / 2.0) / grid_res) * grid_res
            max_lon_grid = math.ceil((center_lon + pad_lon / 2.0) / grid_res) * grid_res
            min_lat_grid = math.floor((center_lat - pad_lat / 2.0) / grid_res) * grid_res
            max_lat_grid = math.ceil((center_lat + pad_lat / 2.0) / grid_res) * grid_res

            # Clamp boundaries to the physical Earth so we don't request Lat 95
            min_lon_grid = max(min_lon_grid, -180.0)
            max_lon_grid = min(max_lon_grid, 180.0)
            min_lat_grid = max(min_lat_grid, -90.0)
            max_lat_grid = min(max_lat_grid, 90.0)

            # Generate the list of BBoxes to fill the current screen
            lon_curr = min_lon_grid
            while lon_curr < max_lon_grid:
                lat_curr = min_lat_grid
                while lat_curr < max_lat_grid:
                    # Append standard [MinLon, MinLat, MaxLon, MaxLat]
                    tile_requests.append([
                        float(lon_curr), 
                        float(lat_curr), 
                        float(lon_curr + grid_res), 
                        float(lat_curr + grid_res)
                    ])
                    lat_curr += grid_res
                lon_curr += grid_res

        # 7. FINAL STATE PACKET
        return {
            "is_tactical": is_tactical_zone,
            # CHANGE THIS LINE: 
            # Instead of cam_speed < 2.0, we use the dwell flag
            "is_pull": is_tactical_zone and self.is_stationary, 
            
            "zoom": zoom_float,
            "center": (center_lat, center_lon),
            "tile_requests": tile_requests, 
            "target_res": "10m" if is_tactical_zone else "GLOBAL"
        }