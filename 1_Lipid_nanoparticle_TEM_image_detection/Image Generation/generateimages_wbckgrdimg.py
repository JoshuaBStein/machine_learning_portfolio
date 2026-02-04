import numpy as np
import cv2
import os
import random
import shutil
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_laplace

# --- CONFIGURATION ---
OUTPUT_DIR = "/scratch/jbs263/LipidDetectionDataset/dataset_1"
CANVAS_SIZE = 1024
NUM_IMAGES_PER_TYPE = 1700 # Adjust as needed (e.g., 500 per type = 3000 total)

# --- UNIFIED CLASS MAP ---
GLOBAL_CLASS_MAP = {
    "solid": 0,
    "ulv": 1,
    "mlv": 2,
    "mvl": 3,
    "bleb": 4
    # Background images will simply have empty label files
}

# --- SHARED UTILITIES ---

def elastic_transform(image, alpha, sigma):
    """Shared deformation logic used by most generators."""
    shape_spatial = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape_spatial) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape_spatial) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape_spatial[1]), np.arange(shape_spatial[0]))
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
    
    if len(image.shape) == 2:
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape_spatial)
    else:
        distorted_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            distorted_image[..., c] = map_coordinates(image[..., c], indices, order=1, mode='reflect').reshape(shape_spatial)
        return distorted_image

def draw_scale_bar(image, nm_length=100, style="black"):
    """Draws a standard 100nm TEM scale bar."""
    h, w = image.shape
    bar_len_px = int(nm_length) # Assuming 1px = 1nm
    
    color = 0 if style == "black" else 255
    
    # Draw line
    cv2.line(image, (w-150, h-50), (w-150+bar_len_px, h-50), color, 8) 
    
    # Draw text
    font_color = color
    cv2.putText(image, "100 nm", (w-150, h-65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, font_color, 2)
    return image

def generate_advanced_debris(size=128):
    """
    Imported from generate_background.py.
    Generates varied debris (blobs, filaments, noise) for realistic background clutter.
    """
    density_map = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    type_debris = random.choice(["blob", "filament", "noise_patch"])
    
    if type_debris == "blob":
        # Ice contamination blob
        cv2.circle(density_map, center, random.randint(5, 20), 0.5, -1)
        density_map = gaussian_filter(density_map, sigma=2)
    elif type_debris == "filament":
        # Carbon edge or filament
        p1 = (random.randint(0, size), random.randint(0, size))
        p2 = (random.randint(0, size), random.randint(0, size))
        cv2.line(density_map, p1, p2, 0.4, 2)
        density_map = gaussian_filter(density_map, sigma=1)
    elif type_debris == "noise_patch":
        noise = np.random.rand(size, size)
        mask = np.random.rand(size, size) > 0.8
        density_map[mask] = noise[mask] * 0.3
    return density_map

# =============================================================================
# GENERATOR 0: BACKGROUND ONLY (New)
# =============================================================================
class GeneratorBackground:
    """
    Generates empty fields with realistic noise and CTF.
    Doesn't generate particles, only physics and background.
    """
    def generate_particle(self):
        # Background generator creates no particles
        return None, None, None

    def apply_physics(self, density_map):
        # Physics model from generate_background.py
        
        # 1. CTF Simulation
        ctf = -(cv2.GaussianBlur(density_map, (0, 0), 4.0) - cv2.GaussianBlur(density_map, (0, 0), 1.5)) * 5.0
        
        # 2. Shot noise
        grain = np.random.normal(0, 0.12, density_map.shape)
        
        # 3. Uneven background illumination
        bg = cv2.resize(np.random.normal(0.5, 0.1, (32, 32)), (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_CUBIC)
        
        final = (bg + ctf + grain)
        
        # Normalize and Clip
        final = (final - np.min(final)) / (np.max(final) - np.min(final) + 1e-5)
        return np.random.poisson(final * 255).astype(np.uint8)

# =============================================================================
# GENERATOR 1: LIPOSOMES (ulv, mlv, mvl)
# =============================================================================
class GeneratorLiposome:
    def draw_thick_membrane(self, size, radius):
        density = np.zeros((size, size), dtype=np.float32)
        center = (size // 2, size // 2)
        thickness = random.uniform(4, 5.5) 
        cv2.circle(density, center, int(radius), 1.0, thickness=int(thickness))
        return gaussian_filter(density, sigma=0.8)

    def generate_particle(self):
        size = 300 
        density_map = np.zeros((size, size), dtype=np.float32)
        binary_mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        
        rand_val = random.random()
        if rand_val < 0.4: morphology = "ulv"
        elif rand_val < 0.7: morphology = "mlv"
        else: morphology = "mvl"

        base_radius = random.randint(35, 85)
        
        # Core Density
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        core_mask = (dist < (base_radius - 2))
        noise = gaussian_filter(np.random.normal(0, 1, (size, size)), sigma=1.5)
        density_map[core_mask] = 0.4 + (noise[core_mask] * 0.15)
        cv2.circle(binary_mask, center, base_radius, 1, -1)

        # Membrane Logic
        if morphology == "ulv":
            density_map += self.draw_thick_membrane(size, base_radius) * 1.5
        elif morphology == "mlv":
            density_map += self.draw_thick_membrane(size, base_radius) * 1.5
            num_layers = random.randint(2, 6)
            curr_rad = base_radius
            for _ in range(num_layers):
                gap = random.randint(9, 15) 
                curr_rad -= gap
                if curr_rad < 10: break
                density_map += self.draw_thick_membrane(size, curr_rad) * 1.2
        elif morphology == "mvl":
            density_map += self.draw_thick_membrane(size, base_radius) * 1.5
            num_vesicles = random.randint(4, 12)
            for _ in range(num_vesicles):
                v_rad = random.randint(8, 20)
                if (base_radius - v_rad - 5) <= 0: continue
                angle = random.uniform(0, 2*np.pi)
                dist_v = random.uniform(0, base_radius - v_rad - 5)
                ox = int(dist_v * np.cos(angle))
                oy = int(dist_v * np.sin(angle))
                v_dens = self.draw_thick_membrane(size, v_rad)
                M = np.float32([[1, 0, ox], [0, 1, oy]])
                density_map += cv2.warpAffine(v_dens, M, (size, size)) * 1.3

        density_map = elastic_transform(density_map, alpha=size*2.5, sigma=size*0.1)
        binary_mask = elastic_transform(binary_mask, alpha=size*2.5, sigma=size*0.1)
        return density_map, binary_mask, morphology

    def apply_physics(self, density_map):
        h, w = density_map.shape
        bg = np.ones((h, w)) * 0.55 
        noise = np.random.normal(0, 1, (h, w))
        bg += ((gaussian_filter(noise, 3.0)*0.4) + (gaussian_filter(noise, 0.5)*0.6)) * 0.05
        canvas = bg - (density_map * 0.25)
        halo = gaussian_filter(density_map, sigma=3.0) * 0.05
        canvas -= halo
        canvas += np.random.normal(0, 0.02, (h, w))
        canvas = (canvas - np.min(canvas)) / (np.max(canvas) - np.min(canvas))
        canvas = (canvas * 200) + 20 
        return np.clip(canvas, 0, 255).astype(np.uint8)

# =============================================================================
# GENERATOR 2: BLEBS
# =============================================================================
class GeneratorBleb:
    def generate_particle(self):
        size = 256
        density_map = np.zeros((size, size), dtype=np.float32)
        binary_mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        y, x = np.ogrid[:size, :size]

        r_body = random.randint(48, 60)
        r_cap = int(r_body * random.uniform(0.75, 0.95))
        offset_dist = (r_body + r_cap) * random.uniform(0.30, 0.55) 
        angle = random.uniform(0, 2 * np.pi)
        cap_cx = center[0] + int(offset_dist * np.cos(angle))
        cap_cy = center[1] + int(offset_dist * np.sin(angle))

        cv2.circle(binary_mask, center, r_body, 1, -1)
        cv2.circle(binary_mask, (int(cap_cx), int(cap_cy)), r_cap, 1, -1)
        
        dx, dy = cap_cx - center[0], cap_cy - center[1]
        dist_centers = np.sqrt(dx*dx + dy*dy)
        if dist_centers > 0:
            nx, ny = -dy/dist_centers, dx/dist_centers
            poly_points = np.array([
                [center[0] + nx * r_body * 0.98, center[1] + ny * r_body * 0.98],
                [center[0] - nx * r_body * 0.98, center[1] - ny * r_body * 0.98],
                [cap_cx - nx * r_cap * 0.98,   cap_cy - ny * r_cap * 0.98],
                [cap_cx + nx * r_cap * 0.98,   cap_cy + ny * r_cap * 0.98]
            ], dtype=np.int32)
            cv2.fillPoly(binary_mask, [poly_points], 1)

        mask_total = (binary_mask > 0)
        dist_cap = np.sqrt((x - cap_cx)**2 + (y - cap_cy)**2)
        mask_phase_A = (dist_cap < r_cap) & mask_total 
        
        density_map[mask_total] = 0.45
        noise_A = cv2.resize(np.random.normal(0,1,(64,64)), (size,size))
        density_map[mask_phase_A] += 0.15 + (noise_A[mask_phase_A] * 0.15)
        density_map[mask_total & (~mask_phase_A)] += 0.02
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(density_map, contours, -1, (0.70), 2)
        
        density_map = gaussian_filter(density_map, sigma=0.8)
        stack = np.dstack((density_map, binary_mask))
        stack = elastic_transform(stack, alpha=size*1.0, sigma=size*0.08)
        return stack[:, :, 0], stack[:, :, 1], "bleb"

    def apply_physics(self, density_map):
        input_soft = gaussian_filter(density_map, sigma=1.5)
        c1 = cv2.GaussianBlur(input_soft, (0, 0), 6.0) 
        c2 = cv2.GaussianBlur(input_soft, (0, 0), 3.0)
        ctf_signal = -(c1 - c2) * 3.0 
        
        bg_low = cv2.resize(np.random.normal(0, 1.0, (64, 64)), density_map.shape[::-1]) * 0.05
        bg_high = np.random.normal(0, 0.06, density_map.shape)
        final = 0.5 + bg_low + bg_high + ctf_signal
        
        final = (final - np.mean(final)) / (np.std(final) + 1e-5) * 0.16 + 0.45
        return (np.clip(final, 0, 1) * 255).astype(np.uint8)

# =============================================================================
# GENERATOR 3: SOLID BASIC
# =============================================================================
class GeneratorSolidBasic:
    def generate_particle(self):
        size = 250 
        density_map = np.zeros((size, size), dtype=np.float32)
        binary_mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        base_radius = random.randint(25, 75) 
        
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        core_noise = cv2.GaussianBlur(np.random.normal(0, 1, (size, size)), (3, 3), 0)

        core_mask = (dist < base_radius)
        binary_mask[core_mask] = 1
        density_map[core_mask] += 0.7 + (core_noise[core_mask] * 0.25)
        
        for _ in range(random.randint(1, 4)):
            void_r = random.randint(3, 8)
            vx, vy = random.randint(center[0]-15, center[0]+15), random.randint(center[1]-15, center[1]+15)
            d_void = np.sqrt((x - vx)**2 + (y - vy)**2)
            density_map[d_void < void_r] -= 0.35

        stack = np.dstack((density_map, binary_mask))
        stack = elastic_transform(stack, alpha=size*1.5, sigma=size*0.08)
        return stack[:, :, 0], stack[:, :, 1], "solid"

    def apply_physics(self, density_map):
        ctf = -(cv2.GaussianBlur(density_map, (0, 0), 4.0) - cv2.GaussianBlur(density_map, (0, 0), 1.5)) * 5.0
        grain = np.random.normal(0, 0.12, density_map.shape)
        bg = cv2.resize(np.random.normal(0.5, 0.1, (32, 32)), (CANVAS_SIZE, CANVAS_SIZE))
        final = (bg + ctf + grain)
        final = (final - np.min(final)) / (np.max(final) - np.min(final) + 1e-5)
        return np.clip(final * 255, 0, 255).astype(np.uint8)

# =============================================================================
# GENERATOR 4: SOLID REALISTIC
# =============================================================================
class GeneratorSolidRealistic:
    def generate_particle(self):
        size = 250 
        density_map = np.zeros((size, size), dtype=np.float32)
        binary_mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        base_radius = random.randint(28, 68) 
        
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        internal_noise = gaussian_filter(np.random.normal(0, 1, (size, size)), sigma=random.uniform(1.5, 2.5))
        core_mask = (dist < base_radius)
        binary_mask[core_mask] = 1
        density_map[core_mask] = 1.0 + (internal_noise[core_mask] * 0.35)
        density_map = gaussian_filter(density_map, sigma=random.uniform(1.5, 2.2))

        stack = np.dstack((density_map, binary_mask))
        stack = elastic_transform(stack, alpha=size*1.2, sigma=size*0.06)
        return stack[:, :, 0], stack[:, :, 1], "solid"

    def apply_physics(self, density_map):
        h, w = density_map.shape
        bias_field = cv2.resize(np.random.randn(64, 64), (w, h), interpolation=cv2.INTER_CUBIC)
        bias_field = gaussian_filter(bias_field, sigma=20) 
        canvas = 0.5 + (bias_field * 0.05)
        canvas -= (density_map * 0.07)
        phase_contrast = gaussian_laplace(density_map, sigma=1.8)
        canvas -= (phase_contrast * 0.008)
        canvas += np.random.normal(0, 0.055, canvas.shape)
        canvas = gaussian_filter(canvas, sigma=0.7)
        canvas = (canvas - np.mean(canvas)) / (np.std(canvas) + 1e-5)
        canvas = (canvas * 32) + 115
        return np.clip(canvas, 0, 255).astype(np.uint8)

# =============================================================================
# GENERATOR 5: SOLID SPHERE
# =============================================================================
class GeneratorSolidSphere:
    def generate_particle(self):
        size = 256
        center = (size // 2, size // 2)
        radius = random.randint(45, 55) 
        y, x = np.ogrid[:size, :size]
        dist_sq = (x - center[0])**2 + (y - center[1])**2
        mask = dist_sq <= radius**2
        
        thickness = np.zeros((size, size), dtype=np.float32)
        thickness[mask] = np.sqrt(radius**2 - dist_sq[mask]) / radius
        thickness = thickness ** 0.8
        return thickness, mask.astype(np.uint8), "solid"

    def apply_physics(self, density_map):
        h, w = density_map.shape
        low_res = np.random.normal(0, 1.0, (h // 16, w // 16))
        mottle = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        grain = np.random.normal(0, 1.0, (h, w))
        bg = 0.72 + (mottle * 0.04) + (grain * 0.03)
        
        img_ideal = bg - (density_map * 0.85)
        blurred = gaussian_filter(img_ideal, sigma=2.0)
        high_pass = img_ideal - blurred
        contrast_enhanced = img_ideal + (high_pass * 4.0)
        final = gaussian_filter(contrast_enhanced, sigma=0.6) + np.random.normal(0, 0.015, (h,w))
        return np.clip(final * 255, 0, 255).astype(np.uint8)

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def create_micrograph(generator, particle_count_range, scale_bar_style="black"):
    full_density = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    occupancy_grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) 
    labels = []
    
    # 1. Place Particles
    num_particles = random.randint(particle_count_range[0], particle_count_range[1])
    
    for _ in range(num_particles):
        p_dens, p_mask, p_type = generator.generate_particle()
        
        # Skip if generator returns None (like Background generator)
        if p_dens is None:
            continue
            
        placed = False
        attempts = 0
        h_p, w_p = p_dens.shape
        
        while not placed and attempts < 50: 
            x_off = random.randint(0, CANVAS_SIZE - w_p)
            y_off = random.randint(0, CANVAS_SIZE - h_p)
            
            existing_slice = occupancy_grid[y_off:y_off+h_p, x_off:x_off+w_p]
            new_footprint = (p_mask > 0.1).astype(np.uint8) 
            
            # Simple overlap check
            if np.sum(existing_slice & new_footprint) > (np.sum(new_footprint) * 0.1):
                attempts += 1 
            else:
                current_slice = full_density[y_off:y_off+h_p, x_off:x_off+w_p]
                full_density[y_off:y_off+h_p, x_off:x_off+w_p] = np.maximum(current_slice, p_dens)
                occupancy_grid[y_off:y_off+h_p, x_off:x_off+w_p] = np.maximum(existing_slice, new_footprint)
                placed = True
                
                # YOLO Label
                coords = cv2.findNonZero((p_mask > 0.5).astype(np.uint8))
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    xc = (x_off + x + w/2)/CANVAS_SIZE
                    yc = (y_off + y + h/2)/CANVAS_SIZE
                    labels.append(f"{GLOBAL_CLASS_MAP[p_type]} {xc:.6f} {yc:.6f} {w/CANVAS_SIZE:.6f} {h/CANVAS_SIZE:.6f}")

    # 2. Add Background Debris (Using the advanced debris from generate_background.py)
    # Applies to all images including background-only
    num_debris = random.randint(10, 30)
    for _ in range(num_debris):
        debris = generate_advanced_debris(size=128)
        
        # Random placement
        h_d, w_d = debris.shape
        x_d, y_d = random.randint(0, CANVAS_SIZE-w_d), random.randint(0, CANVAS_SIZE-h_d)
        
        current_slice = full_density[y_d:y_d+h_d, x_d:x_d+w_d]
        full_density[y_d:y_d+h_d, x_d:x_d+w_d] = np.maximum(current_slice, debris)

    # 3. Apply Physics (specific to the generator class)
    final_img = generator.apply_physics(full_density)
    
    # 4. Draw Scale Bar
    final_img = draw_scale_bar(final_img, style=scale_bar_style)
        
    return final_img, labels

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning previous output at {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
        
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)
    
    with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
        f.write("solid\nulv\nmlv\nmvl\nbleb")

    # Define tasks: (Prefix, GeneratorInstance, ParticleCount, ScaleBarStyle)
    tasks = [
        ("liposome", GeneratorLiposome(), (30, 50), "black"),
        ("bleb", GeneratorBleb(), (40, 60), "white"),
        ("solid_calibrated", GeneratorSolidBasic(), (20, 60), "white"),
        ("solid_realistic", GeneratorSolidRealistic(), (35, 55), "white"), 
        ("solid_sphere", GeneratorSolidSphere(), (15, 22), "black"),
        # New Background Task: 0 particles, black scale bar
        ("background_noise", GeneratorBackground(), (0, 0), "black") 
    ]

    print(f"--- Starting Master Generation ({NUM_IMAGES_PER_TYPE} images per type) ---")
    print(f"Output Directory: {OUTPUT_DIR}")

    for prefix, generator, p_range, bar_style in tasks:
        print(f"Generating set: {prefix}...")
        for i in range(NUM_IMAGES_PER_TYPE):
            img, lbls = create_micrograph(generator, p_range, bar_style)
            
            filename = f"{prefix}_{i:04d}"
            cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.png", img)
            
            # Write label file (empty for background)
            with open(f"{OUTPUT_DIR}/labels/{filename}.txt", "w") as f:
                f.write("\n".join(lbls))
                
            if i % 100 == 0 and i > 0:
                print(f"  {prefix}: {i}/{NUM_IMAGES_PER_TYPE} done")

    print(f"All done! Dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
