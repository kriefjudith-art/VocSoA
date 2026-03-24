import numpy as np
import sounddevice as sd
import pygame
import sys
import math

# --- OPTIMIZED AUDIO SETTINGS ---
# Lowered sample rate heavily reduces latency and CPU load for YIN
SAMPLE_RATE = 22050 
BUFFER_SIZE = 1024 
MIN_PITCH = 150.0  
MAX_PITCH = 800.0  
APERIODICITY_THRESHOLD = 0.2  

# Shared buffer and flag for the audio thread
audio_data = np.zeros(BUFFER_SIZE)
new_audio_ready = False

def audio_callback(indata, frames, time, status):
    """ Captures live audio and flags that new data is ready """
    global audio_data, new_audio_ready
    if status:
        print(f"Audio status: {status}")
    audio_data = indata[:, 0]
    new_audio_ready = True

def yin_pitch_and_aperiodicity(sig, sr, f0_min=100, f0_max=1000):
    """ Fast NumPy YIN implementation """
    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)
    
    W = len(sig) - tau_max
    if W <= 0:
        return -1, 1.0

    df = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff = sig[:W] - sig[tau:W+tau]
        df[tau] = np.sum(diff ** 2)
        
    cmndf = np.zeros(tau_max)
    cmndf[0] = 1.0
    running_sum = 0.0
    
    for tau in range(1, tau_max):
        running_sum += df[tau]
        cmndf[tau] = df[tau] * tau / (running_sum + 1e-8) 
        
    tau_estimate = -1
    aperiodicity = 1.0
    
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < APERIODICITY_THRESHOLD:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            tau_estimate = tau
            aperiodicity = cmndf[tau]
            break
            
    if tau_estimate == -1:
        return -1, 1.0
        
    pitch = sr / tau_estimate
    return pitch, aperiodicity

def draw_cooler_chick(surface, x, y, angle, time_ms, is_flying):
    """ Draws an animated, cool superhero chick """
    chick_surface = pygame.Surface((120, 120), pygame.SRCALPHA)
    
    # 1. Draw Cape (Flaps if flying)
    cape_flap = math.sin(time_ms * 0.02) * 10 if is_flying else 0
    pygame.draw.polygon(chick_surface, (220, 20, 60), [(20, 40), (40, 60), (10, 80 + cape_flap)])
    
    # 2. Draw Body (Bright Yellow)
    pygame.draw.circle(chick_surface, (255, 215, 0), (60, 60), 40)
    
    # 3. Draw Animated Wing
    # Wing goes up and down based on time
    wing_y = math.sin(time_ms * 0.015) * 15 if is_flying else 0
    pygame.draw.ellipse(chick_surface, (255, 165, 0), (35, 55 + wing_y, 40, 20))
    
    # 4. Draw Cool Sunglasses
    pygame.draw.line(chick_surface, (0, 0, 0), (65, 45), (95, 45), 4) # Bridge
    pygame.draw.circle(chick_surface, (30, 30, 30), (70, 45), 10) # Left lens
    pygame.draw.circle(chick_surface, (30, 30, 30), (90, 45), 10) # Right lens
    # Little white reflections on the glasses
    pygame.draw.circle(chick_surface, (255, 255, 255), (67, 42), 3) 
    pygame.draw.circle(chick_surface, (255, 255, 255), (87, 42), 3) 

    # 5. Draw Beak
    pygame.draw.polygon(chick_surface, (255, 140, 0), [(95, 55), (115, 65), (95, 75)])
    
    # Rotate and blit to main screen
    rotated_chick = pygame.transform.rotate(chick_surface, angle)
    rect = rotated_chick.get_rect(center=(int(x), int(y)))
    surface.blit(rotated_chick, rect.topleft)

def main():
    global new_audio_ready
    pygame.init()
    
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Baby Pitch Agency Study - Low Latency")
    clock = pygame.time.Clock()

    bird_x = 100.0
    bird_y = height / 2.0
    bird_angle = 0.0
    target_y = height / 2.0
    trajectory = []
    
    # Store last known valid pitch to prevent jitter
    current_pitch = -1 
    is_vocalizing = False

    # Force 'low' latency request from the sound device
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                            callback=audio_callback, blocksize=BUFFER_SIZE, 
                            latency='low')
    
    with stream:
        running = True
        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # --- PROCESS AUDIO (ONLY IF NEW DATA ARRIVED) ---
            if new_audio_ready:
                audio_copy = np.copy(audio_data)
                new_audio_ready = False # Reset flag immediately
                
                pitch, aperiodicity = yin_pitch_and_aperiodicity(audio_copy, SAMPLE_RATE, MIN_PITCH, MAX_PITCH)
                
                if pitch != -1 and aperiodicity < APERIODICITY_THRESHOLD:
                    if MIN_PITCH <= pitch <= MAX_PITCH:
                        current_pitch = pitch
                        is_vocalizing = True
                    else:
                        is_vocalizing = False
                else:
                    is_vocalizing = False

            # --- UPDATE STATE ---
            if is_vocalizing:
                bird_x += 6.0 # Slightly faster
                
                normalized_pitch = (current_pitch - MIN_PITCH) / (MAX_PITCH - MIN_PITCH)
                normalized_pitch = max(0.1, min(0.9, normalized_pitch))
                target_y = height - (normalized_pitch * height)
                
                trajectory.append((int(bird_x), int(bird_y)))

            # Smooth vertical movement
            dy = target_y - bird_y
            bird_y += dy * 0.15

            target_angle = (dy * -0.05) if is_vocalizing else 0.0
            bird_angle += (target_angle - bird_angle) * 0.2

            if bird_x > width + 50:
                bird_x = 0
                trajectory.clear()

            # --- DRAWING ---
            screen.fill((135, 206, 235)) 

            if len(trajectory) > 1:
                pygame.draw.lines(screen, (255, 20, 147), False, trajectory, 16)
                pygame.draw.lines(screen, (255, 255, 255), False, trajectory, 6)

            # Draw the upgraded cool bird
            draw_cooler_chick(screen, bird_x, bird_y, bird_angle, current_time, is_vocalizing)

            pygame.display.flip()
            clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()