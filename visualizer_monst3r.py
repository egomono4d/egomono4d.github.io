import time
import sys
import argparse
from pathlib import Path

import numpy as onp
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf
import matplotlib.cm as cm  # For colormap
import argparse
import pickle
import time
import cv2
import copy
import imageio 
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import viser
import viser.transforms as tf


def downsample(position, color, flys, downsample_factor):
    new_height = position.shape[0] // downsample_factor
    new_width = position.shape[1] // downsample_factor
    position_downsampled = cv2.resize(position, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    color_downsampled = cv2.resize(color, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    flys_downsampled = cv2.resize(flys, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    flys_downsampled = np.round(flys_downsampled).astype(np.float32)

    return position_downsampled, color_downsampled, flys_downsampled


def main(
    file_name, save_fp, xyzs, rgbs, flys, intrinsics, extrinsics,
    point_size=0.01,
    camera_frustum_scale=0.2,
    axes_scale=0.1,
    downsample_factor=1,
    cam_thickness=1.5,
) -> None:
    from pathlib import Path  # <-- Import Path here if not already imported
    server = viser.ViserServer()

    server.scene.set_up_direction('-z')
    
    num_frames = len(xyzs)
    print(f"Number of frames: {num_frames}")


    fps = 10 if pickle_file.startswith("hoi4d_") else 20

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,  # Initially disabled
        )

    # Add recording UI.
    with server.gui.add_folder("Recording"):
        gui_record_scene = server.gui.add_button("Record Scene")

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider
        if gui_show_all_frames.value:
            # Show frames with stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)
            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            # Show only the current frame
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == current_timestep
            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Update frame visibility based on new stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)

    # Recording handler
    @gui_record_scene.on_click
    def _(_):
        gui_record_scene.disabled = True

        # Save the original frame visibility state
        original_visibility = [frame_node.visible for frame_node in frame_nodes]

        rec = server._start_scene_recording()
        rec.set_loop_start()
        
        # Determine sleep duration based on current FPS
        sleep_duration = 1.0 / gui_framerate.value if gui_framerate.value > 0 else 0.033  # Default to ~30 FPS
        
        if gui_show_all_frames.value:
            # Record all frames according to the stride
            stride = gui_stride.value
            frames_to_record = [i for i in range(num_frames) if i % stride == 0]
        else:
            # Record the frames in sequence
            frames_to_record = range(num_frames)
        
        for t in frames_to_record:
            # Update the scene to show frame t
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i == t) if not gui_show_all_frames.value else (i % gui_stride.value == 0)
            server.flush()
            rec.insert_sleep(sleep_duration)

        # set all invisible
        with server.atomic():
            for frame_node in frame_nodes:
                frame_node.visible = False
        
        # Finish recording
        bs = rec.end_and_serialize()
        
        # Save the recording to a file
        output_path = Path(os.path.join(save_fp, file_name+".viser"))
        # make sure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bs)
        print(f"Recording saved to {output_path.resolve()}")
        
        # Restore the original frame visibility state
        with server.atomic():
            for frame_node, visibility in zip(frame_nodes, original_visibility):
                frame_node.visible = visibility
        server.flush()
        
        gui_record_scene.disabled = False

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        position = xyzs[i]       # (H, W, 3)
        color = rgbs[i]          # (H, W, 3)
        fly = flys[i]           # (H, W)
        position, color, fly = downsample(position, color, fly, downsample_factor)
        position_valid = position.reshape(-1, 3)
        color_valid = color.reshape(-1, 3)

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position_valid,
            colors=color_valid,
            point_size=point_size,
            point_shape="rounded",
        )

        # Compute color for frustum based on frame index.
        norm_i = i / (num_frames - 1) if num_frames > 1 else 0  # Normalize index to [0, 1]
        color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components

        # Place the frustum with the computed color.
        fov = 2 * np.arctan2(rgbs[i].shape[0] / 2, intrinsics[i, 0, 0] * rgbs[i].shape[1])
        aspect = rgbs[i].shape[1] / rgbs[i].shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=camera_frustum_scale,
            image=rgbs[i][::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(extrinsics[i][:3, :3]).wxyz,
            position=extrinsics[i][:3, 3],
            color=color_rgb,  # Set the color for the frustum
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 10,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    # Initialize frame visibility.
    for i, frame_node in enumerate(frame_nodes):
        if gui_show_all_frames.value:
            frame_node.visible = (i % gui_stride.value == 0)
        else:
            frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value and not gui_show_all_frames.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":

    idx = 3

    fp_list = os.listdir('.')
    fp_list = [x for x in fp_list if x.endswith(".pkl")]

    pickle_file = fp_list[idx]

    if True:

        print(f"{idx}: {pickle_file}")


        with open(pickle_file, 'rb') as f:
            result = pickle.load(f)
        xyzs, rgbs, flys, intrinsics, extrinsics = result['xyzs'], result['rgbs'], result['flys'], result['intrinsics'], result['extrinsics']
        pickle_file = pickle_file[:-4]

        downsample_factor = 1
            
        rgbs_video_usage = copy.deepcopy(rgbs)
        new_height = rgbs_video_usage.shape[1] // downsample_factor
        new_width = rgbs_video_usage.shape[2] // downsample_factor
        
        video_fp = os.path.join("recordings", pickle_file+".mp4")
        if pickle_file.startswith("hoi4d_"):
            writer = imageio.get_writer(video_fp, fps=10)
        else:
            writer = imageio.get_writer(video_fp, fps=20)
        
        for i in range(rgbs_video_usage.shape[0]):
            frame = cv2.resize(rgbs_video_usage[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)
            if i == 0:
                img_fp = os.path.join("recordings", pickle_file+".jpg")
                cv2.imwrite(img_fp, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.close()


        # Call the main function with the parsed arguments
        tyro.cli(main(
            pickle_file, "recordings", xyzs, rgbs, flys, intrinsics, extrinsics,
            point_size=0.01,
            camera_frustum_scale=0.2,
            axes_scale=0.1,
            downsample_factor=1,
            cam_thickness=1.5,
        ))
