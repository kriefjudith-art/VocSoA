import numpy as np
import sounddevice as sd
import pygame
import sys

# --- AUDIO & STUDY SETTINGS ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048 
MIN_PITCH = 150.0  # Lowest expected baby grunt
MAX_PITCH = 800.0  # Highest expected baby squeal
APERIODICITY_THRESHOLD = 0.2  # Lower means stricter harmonicity (0.1 to 0.3 is typical for voice)

# Shared buffer for the audio thread
audio_data = np.zeros(BUFFER_SIZE)

def audio_callback(indata, frames, time, status):
    """ Captures live audio from the microphone into a global buffer """
    global audio_data
    if status:
        print(f"Audio status: {status}")
    # Grab the first channel (mono)
    audio_data = indata[:, 0]

def yin_pitch_and_aperiodicity(sig, sr, f0_min=100, f0_max=1000):
    """
    A pure numpy implementation of the YIN algorithm.
    Returns: (pitch_hz, aperiodicity)
    """
    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)
    
    # Window size must be large enough to contain the lowest frequency
    W = len(sig) - tau_max
    if W <= 0:
        return -1, 1.0

    # Step 1: Difference function
    df = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff = sig[:W] - sig[tau:W+tau]
        df[tau] = np.sum(diff ** 2)
        
    # Step 2: Cumulative mean normalized difference function (CMNDF)
    cmndf = np.zeros(tau_max)
    cmndf[0] = 1.0
    running_sum = 0.0
    
    for tau in range(1, tau_max):
        running_sum += df[tau]
        cmndf[tau] = df[tau] * tau / (running_sum + 1e-8) # Add tiny value to avoid div by zero
        
    # Step 3: Absolute threshold
    tau_estimate = -1
    aperiodicity = 1.0
    
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < APERIODICITY_THRESHOLD:
            # Find the local minimum
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            tau_estimate = tau
            aperiodicity = cmndf[tau]
            break
            
    # If no dip is found below the threshold, the sound is likely unvoiced/noise
    if tau_estimate == -1:
        return -1, 1.0
        
    # Calculate pitch from the period (tau)
    pitch = sr / tau_estimate
    return pitch, aperiodicity

def draw_chick(surface, x, y, angle):
    """ Draws a cute, high-contrast baby chick """
    # Create a temporary surface to handle rotation
    chick_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
    
    # Draw body (Bright Yellow)
    pygame.draw.circle(chick_surface, (255, 215, 0), (50, 50), 40)
    # Draw eye (Black)
    pygame.draw.circle(chick_surface, (0, 0, 0), (65, 35), 8)
    # Draw beak (Orange)
    pygame.draw.polygon(chick_surface, (255, 140, 0), [(85, 45), (105, 55), (85, 65)])
    
    # Rotate and blit
    rotated_chick = pygame.transform.rotate(chick_surface, angle)
    rect = rotated_chick.get_rect(center=(int(x), int(y)))
    surface.blit(rotated_chick, rect.topleft)

def main():
    pygame.init()
    
    # Set up the display (Fullscreen is often best for babies, but we use windowed here for easy testing)
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Baby Pitch Agency Study")
    clock = pygame.time.Clock()

    # Game State
    bird_x = 100.0
    bird_y = height / 2.0
    bird_angle = 0.0
    target_y = height / 2.0
    trajectory = []

    # Start audio stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                            callback=audio_callback, blocksize=BUFFER_SIZE)
    
    with stream:
        running = True
        while running:
            # Handle Pygame events (FIXED)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # --- PROCESS AUDIO ---
            # Copy buffer to avoid thread collisions while processing
            current_audio = np.copy(audio_data)
            pitch, aperiodicity = yin_pitch_and_aperiodicity(current_audio, SAMPLE_RATE, MIN_PITCH, MAX_PITCH)
            
            is_vocalizing = False

            # --- UPDATE STATE ---
            # Trigger ONLY if the sound is highly harmonic AND within the baby's pitch range
            if pitch != -1 and aperiodicity < APERIODICITY_THRESHOLD:
                if MIN_PITCH <= pitch <= MAX_PITCH:
                    is_vocalizing = True
                    
                    # Advance the bird
                    bird_x += 5.0
                    
                    # Map pitch to Y axis (high pitch = top of screen)
                    normalized_pitch = (pitch - MIN_PITCH) / (MAX_PITCH - MIN_PITCH)
                    normalized_pitch = max(0.1, min(0.9, normalized_pitch))
                    target_y = height - (normalized_pitch * height)
                    
                    trajectory.append((int(bird_x), int(bird_y)))

            # Smooth vertical movement (Linear Interpolation)
            dy = target_y - bird_y
            bird_y += dy * 0.15

            # Calculate tilt angle (Pygame rotation is counter-clockwise, so we invert dy)
            target_angle = (dy * -0.05) if is_vocalizing else 0.0
            bird_angle += (target_angle - bird_angle) * 0.2

            # Screen wrapping
            if bird_x > width + 50:
                bird_x = 0
                trajectory.clear()

            # --- DRAWING ---
            screen.fill((135, 206, 235)) # Sky Blue background

            # Draw the candy-striped trajectory line
            if len(trajectory) > 1:
                # Outer hot pink glow
                pygame.draw.lines(screen, (255, 20, 147), False, trajectory, 16)
                # Inner white line
                pygame.draw.lines(screen, (255, 255, 255), False, trajectory, 6)

            # Draw the chick
            draw_chick(screen, bird_x, bird_y, bird_angle)

            # Update display
            pygame.display.flip()
            
            # Run at roughly 60 FPS
            clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()