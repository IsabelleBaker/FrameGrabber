import os, cv2, wx, math
import pickle
import numpy as np
import torch
import torchvision


class FrameGrabberInitialWindow(wx.Frame):

    def __init__(self, title):
        wx.Frame.__init__(self, parent=None, title=title)
        self.control_panel = MyControlsPanel(self)
        self.video_panel = MyVideoPanel(self)
        self.frame_sizer = wx.BoxSizer(wx.VERTICAL)
        self.frame_sizer.Add(self.control_panel, 0, wx.EXPAND)
        self.frame_sizer.Add(self.video_panel, 1, wx.EXPAND)
        self.SetSizer(self.frame_sizer)
        self.Size = (self.control_panel.BestVirtualSize[0] + self.video_panel.BestVirtualSize[0],
                     self.control_panel.BestVirtualSize[1] + self.video_panel.BestVirtualSize[1])
        self.Move(wx.Point(50, 50))
        self.Show()


class MyVideoPanel(wx.ScrolledWindow):

    def __init__(self, parent):
        wx.ScrolledWindow.__init__(self, parent, id=-1, pos=wx.DefaultPosition, size=(100, 400),
                                   style=wx.HSCROLL | wx.VSCROLL,
                                   name="scrolledWindow")
        self.SetScrollbars(1, 1, 10, 10)
        self.parent = parent
        self.frame = np.zeros(shape=(3, 3, 3))
        self.Bind(wx.EVT_PAINT, self.evt_on_paint)
        self.Bind(wx.EVT_SIZE, self.evt_on_resize)
        self.Image = wx.StaticBitmap(self, -1)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.Image, 1, wx.EXPAND | wx.ALL)

    def evt_on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        bmp = wx.Bitmap.FromBuffer(self.frame.shape[1], self.frame.shape[0],
                                   cv2.resize(self.frame, (self.frame.shape[1], self.frame.shape[0])))
        dc.DrawBitmap(bmp, 0, 0, False)

    def evt_on_resize(self, event):
        if self.parent.control_panel.cap:
            current_position = self.parent.control_panel.video_slider.GetValue()
            self.parent.control_panel.update_display(current_position)

class MyControlsPanel(wx.ScrolledWindow):
    def __init__(self, parent):
        wx.ScrolledWindow.__init__(self, parent, id=-1, pos=wx.DefaultPosition, size=(200, 260),
                                   style=wx.HSCROLL | wx.VSCROLL,
                                   name="scrolledWindow")
        self.animal_tracking_dictionary = None
        self.pickle_path = None
        self.SetScrollbars(1, 1, 600, 400)
        # Set up the variables that we want to capture
        self.parent = parent
        self.output_path = None
        self.video_path = None
        self.cap = None
        self.video_length = 0
        self.output_size = 1280

        # Make the components for the app and assign their bindings
        get_video_button = wx.Button(self, label='Select A Video File')
        get_video_button.SetToolTip('Select the video that matches your animal tracks')
        get_video_button.Bind(wx.EVT_BUTTON, self.evt_get_video_path)
        self.get_video_label = wx.TextCtrl(self, value='', style=wx.TE_LEFT, size=(300, -1))
        self.get_video_label.SetHint('{your video}')
        self.output_directory_button = wx.Button(self, label='Output Path')
        self.output_directory_button.SetToolTip('Location to store images')
        self.output_directory_button.Bind(wx.EVT_BUTTON, self.evt_get_output_directory)
        self.output_directory_label = wx.TextCtrl(self, value='',
                                                  style=wx.TE_LEFT, size=(300, -1))
        self.output_directory_label.SetHint('{output directory}')



        self.overlay_track_checkbox = wx.CheckBox(self, label='Overlay Track')  # Create the checkbox
        self.overlay_track_checkbox.SetValue(False)  # Set the default to "checked"
        self.overlay_track_checkbox.Bind(wx.EVT_CHECKBOX, self.evt_overlay_track_checkbox)

        self.get_tracklet_directory_button = wx.Button(self, label='Track File Path')
        self.get_tracklet_directory_button.SetToolTip('Optional: Location of pickle file containing track information')
        self.get_tracklet_directory_button.Bind(wx.EVT_BUTTON, self.evt_get_tracklet)
        self.get_tracklet_directory_label = wx.TextCtrl(self, value='',
                                                  style=wx.TE_LEFT, size=(300, -1))
        self.get_tracklet_directory_label.SetHint('{optional: track file}')

        self.resize_checkbox = wx.CheckBox(self, label='Resize Output')  # Create the checkbox
        self.resize_checkbox.SetValue(False)  # Set the default to "checked"
        self.resize_checkbox.Bind(wx.EVT_CHECKBOX, self.evt_resize_checkbox)
        self.resize_widget = wx.SpinCtrlDouble(self, initial=1280, min=32, max=12800, inc=32)
        wx.Button.SetToolTip(self.resize_widget, 'Select the size (in pixels) that you would like '
                                                 'for your image output. A "padded" square '
                                                 'image will be saved.')
        self.resize_widget.Bind(wx.EVT_SPINCTRLDOUBLE, self.evt_set_resize)
        self.output_size = int(self.resize_widget.GetValue())
        self.video_slider = wx.Slider(self, name='Video Scroll Bar', style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.video_slider.SetToolTip('Scroll through the video frames')
        self.video_slider.Bind(wx.EVT_SCROLL_THUMBTRACK, self.evt_scrolling)
        self.backward_button = wx.Button(self, label='<<< step')
        self.backward_button.SetToolTip('Step Backward a single frame in the video')
        self.backward_button.Bind(wx.EVT_BUTTON, self.evt_step_backward)
        self.forward_button = wx.Button(self, label='step >>>')
        self.forward_button.SetToolTip('Step Forward a single frame in the video')
        self.forward_button.Bind(wx.EVT_BUTTON, self.evt_step_forward)
        self.save_frame_button = wx.Button(self, label='Save Frame')
        self.save_frame_button.SetToolTip('Save the currently displayed frame to output folder')
        self.save_frame_button.Bind(wx.EVT_BUTTON, self.evt_save_frame)

        # Place everything on the page
        overall_window_vertical = wx.BoxSizer(wx.VERTICAL)
        overall_window_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        get_video_sizer_vertical = wx.StaticBox(self)
        get_video_vertical = wx.StaticBoxSizer(get_video_sizer_vertical, wx.VERTICAL)
        get_video_options = wx.BoxSizer(wx.HORIZONTAL)
        get_video_options.Add(get_video_button, 0, flag=wx.ALIGN_CENTER)
        get_video_options.Add(10, 0)
        get_video_options.Add(self.get_video_label, 0, flag=wx.ALIGN_CENTER)
        get_video_vertical.Add(0, 5)
        get_video_vertical.Add(get_video_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)
        frame_output = wx.BoxSizer(wx.HORIZONTAL)
        frame_output.Add(self.output_directory_button, flag=wx.ALIGN_CENTER)
        frame_output.Add(10, 0)
        frame_output.Add(self.output_directory_label, flag=wx.ALIGN_CENTER)
        get_video_vertical.Add(frame_output, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)
        overall_window_vertical.Add(get_video_vertical, flag=wx.EXPAND)

        overlay_track_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        overlay_track_box = wx.StaticBox(self)
        overlay_track_options_vertical_sizer = wx.StaticBoxSizer(overlay_track_box, wx.VERTICAL)
        overlay_track_parts_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        overlay_track_parts_horizontal.Add(self.overlay_track_checkbox, wx.CENTER)
        overlay_track_parts_horizontal.Add(10, 0)
        overlay_track_parts_horizontal.Add(self.get_tracklet_directory_button, 0, flag=wx.ALIGN_CENTER)
        overlay_track_parts_horizontal.Add(10, 0)
        overlay_track_parts_horizontal.Add(self.get_tracklet_directory_label, 0, flag=wx.ALIGN_CENTER)
        overlay_track_options_vertical_sizer.Add(overlay_track_parts_horizontal, wx.LEFT)
        overlay_track_horizontal.Add(overlay_track_options_vertical_sizer, wx.ALIGN_CENTER_HORIZONTAL)
        overall_window_vertical.Add(overlay_track_horizontal, flag=wx.EXPAND)

        resize_video_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        resize_video_box = wx.StaticBox(self)
        resize_video_options_vertical_sizer = wx.StaticBoxSizer(resize_video_box, wx.VERTICAL)
        resize_video_parts_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        resize_video_parts_horizontal.Add(self.resize_checkbox, wx.CENTER)
        resize_video_parts_horizontal.Add(10, 0)
        resize_video_parts_horizontal.Add(self.resize_widget, wx.CENTER)
        resize_video_options_vertical_sizer.Add(resize_video_parts_horizontal, wx.LEFT)
        resize_video_horizontal.Add(resize_video_options_vertical_sizer, wx.ALIGN_CENTER_HORIZONTAL)
        overall_window_vertical.Add(resize_video_horizontal, flag=wx.EXPAND)
        play_video_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        play_video_box = wx.StaticBox(self)
        play_video_options_vertical_sizer = wx.StaticBoxSizer(play_video_box, wx.VERTICAL)
        play_video_parts_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        play_video_parts_horizontal.Add(self.backward_button, wx.CENTER)
        play_video_parts_horizontal.Add(10, 0)
        play_video_parts_horizontal.Add(self.forward_button, wx.CENTER)
        play_video_parts_horizontal.Add(10, 0)
        play_video_parts_horizontal.Add(self.save_frame_button, wx.CENTER)
        play_video_options_vertical_sizer.Add(play_video_parts_horizontal, wx.LEFT)
        play_video_horizontal.Add(play_video_options_vertical_sizer, wx.ALIGN_CENTER_HORIZONTAL)
        overall_window_vertical.Add(play_video_horizontal, flag=wx.EXPAND)
        overall_window_vertical.Add(self.video_slider, wx.ALIGN_CENTER_HORIZONTAL, wx.EXPAND)
        overall_window_vertical.Add(0, 15)
        overall_window_horizontal.Add(15, 0)
        overall_window_horizontal.Add(overall_window_vertical, wx.EXPAND)
        overall_window_horizontal.Add(15, 0)
        self.SetSizer(overall_window_horizontal)

        # disable buttons
        self.video_slider.Disable()
        self.backward_button.Disable()
        self.forward_button.Disable()
        self.save_frame_button.Disable()
        self.overlay_track_checkbox.Disable()
        self.get_tracklet_directory_button.Disable()
        self.resize_widget.Disable()
        self.resize_checkbox.Disable()
        self.output_directory_button.Disable()

    def evt_scrolling(self, event):
        self.update_display(self.video_slider.GetValue())

    def evt_step_forward(self, event):
        self.video_slider.SetValue(self.video_slider.GetValue() + 1)
        self.update_display(self.video_slider.GetValue())

    def evt_step_backward(self, event):
        self.video_slider.SetValue(self.video_slider.GetValue() - 1)
        self.update_display(self.video_slider.GetValue())

    def evt_resize_checkbox(self, event):
        if self.resize_checkbox.GetValue():
            self.resize_widget.Enable()
        else:
            self.resize_widget.Disable()

    def evt_overlay_track_checkbox(self, event):
        if self.overlay_track_checkbox.GetValue():
            self.get_tracklet_directory_button.Enable()
        else:
            self.get_tracklet_directory_button.Disable()
    def evt_set_resize(self, event):
        self.output_size = int(self.resize_widget.GetValue())

    def evt_get_video_path(self, event):
        wildcard = "Videos (*.mp4, *.mov, *.avi)|*.mp4;*.mov;*.avi"
        dlg = wx.FileDialog(
            self, message="Choose a Video",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.video_slider.Enable()
            self.backward_button.Enable()
            self.forward_button.Enable()
            self.resize_checkbox.Enable()
            self.output_directory_button.Enable()
            self.overlay_track_checkbox.Enable()
            self.video_path = path
            self.get_video_label.SetValue(os.path.basename(path))
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            self.queue_depth = 100 if self.video_length >= 100 else self.video_length
            self.video_slider.SetRange(0, self.video_length)
            self.video_slider.SetValue(0)
            self.update_display(self.video_slider.GetValue())
        dlg.Destroy()

    def evt_get_tracklet(self, event):
        wildcard = "Track Files (*.pkl)|*.pkl"
        dlg = wx.FileDialog(
            self, message="Choose a Track File",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'rb') as f:
                temp_pickle = pickle.load(f)
            if os.path.basename(self.video_path) != temp_pickle['video_info']['filename']:
                dlg = wx.GenericMessageDialog(None, "Track file name does not match video file!", caption='Error',
                                              style=wx.OK | wx.CENTER)
                dlg.ShowModal()
                return
            self.pickle_path = path
            self.get_tracklet_directory_label.SetValue(os.path.basename(path))
            self.animal_tracking_dictionary = temp_pickle
        dlg.Destroy()

    def update_display(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        _, display_frame = self.cap.read()
        if display_frame is None:
            dlg = wx.GenericMessageDialog(None, 'It appears your file metadata or video is corrupted. Finding last valid frame and Adjusting range', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            # Because the slider doesn't necessarily increment by 1,
            # need to loop backward to find last valid frame then set the range
            while display_frame is None:
                position -= 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                _, display_frame = self.cap.read()
            self.video_slider.SetValue(position)
            self.video_slider.SetRange(self.video_slider.GetRange()[0], position)
        display_size = self.parent.video_panel.GetSize()
        if display_size.GetWidth() <= 0 or display_size.GetHeight() <= 0: return
        if self.parent.control_panel.animal_tracking_dictionary:
            display_frame = self.draw_frame(self.video_slider.GetValue(), display_frame)
        display_frame = self.resize_frame(display_frame, display_size.GetWidth(), display_size.GetHeight())
        self.parent.video_panel.frame = np.transpose(display_frame, (1, 2, 0))
        self.parent.video_panel.Refresh()

    def evt_get_output_directory(self, event):
        dlg = wx.DirDialog(None, "Choose output directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.output_path = dlg.GetPath()
            self.output_directory_label.LabelText = self.output_path
            self.save_frame_button.Enable()
        dlg.Destroy()

    def evt_save_frame(self, event):
        filename_original = os.path.join(self.output_path,
                                         f'{os.path.basename(self.video_path)[:-4]}_'
                                         f'{self.video_slider.GetValue()}.jpg')
        current_position = self.video_slider.GetValue()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        _, frame = self.cap.read()
        if self.resize_checkbox.GetValue():
            frame = self.resize_frame(frame, self.output_size, self.output_size)
            frame = np.transpose(frame, (1, 2, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_original, frame)
        return

    def resize_frame(self, frame, display_width, display_height):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_to_tensor = torchvision.transforms.ToTensor()(img_rgb)

        # Convert the image to a tensor and scale it
        img_to_tensor = img_to_tensor * 255

        # Make the tensor an uint8 type
        frame = img_to_tensor.type(torch.uint8)
        pad = (0, 0, 0, 0)

        # Resize the image to feed into the model by padding it to make it square and divisible by 32.
        # Then resize to the output size.
        frame_width = frame.shape[2]
        frame_height = frame.shape[1]
        if int((frame_width / frame_height) * display_height) * int((frame_height / frame_width) * display_width) == 0:
            resize = torchvision.transforms.Resize(size=1,
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                   antialias=True)
        elif frame_width == frame_height:
            resize = torchvision.transforms.Resize(size=display_height,
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                   antialias=True)
        elif display_width > display_height:
            if frame_width > frame_height:
                resize = torchvision.transforms.Resize(size=display_height, max_size=display_width,
                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                       antialias=True)
            if frame_width < frame_height:
                resize = torchvision.transforms.Resize(size=int((frame_width / frame_height) * display_height),
                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                       antialias=True)
        elif display_width < display_height:
            if frame_width < frame_height:
                resize = torchvision.transforms.Resize(size=display_width, max_size=display_height,
                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                       antialias=True)
            if frame_width > frame_height:
                resize = torchvision.transforms.Resize(size=int((frame_height / frame_width) * display_width),
                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                       antialias=True)
        else:
            if frame_width < frame_height:
                resize = torchvision.transforms.Resize(size=int((frame_width / frame_height) * display_height),
                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                       antialias=True)
            if frame_width > frame_height:
                resize = torchvision.transforms.Resize(size=int((frame_height / frame_width) * display_width),
                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                       antialias=True)
        processed_frame = resize(frame)
        if processed_frame.shape[2] == display_width:
            bottom_pad = math.floor((display_height - processed_frame.shape[1]) / 2)
            top_pad = display_height - processed_frame.shape[1] - bottom_pad
            right_pad = 0
            left_pad = 0

        else:
            right_pad = math.floor((display_width - processed_frame.shape[2]) / 2)
            left_pad = display_width - processed_frame.shape[2] - right_pad
            bottom_pad = math.floor((display_height - processed_frame.shape[1]) / 2)
            top_pad = display_height - processed_frame.shape[1] - bottom_pad
        pad = (left_pad, right_pad, top_pad, bottom_pad)
        processed_frame = torch.nn.functional.pad(processed_frame, pad)
        return_frame = processed_frame.numpy()
        return return_frame

    def draw_frame(self, frame_number, frame=None):
        frame_temp = frame
        for animal in self.animal_tracking_dictionary['animals']:
            if frame_number in self.animal_tracking_dictionary['animals'][animal]['frame_number']:
                frame_index = self.animal_tracking_dictionary['animals'][animal]['frame_number'].index(frame_number)
                color = self.animal_tracking_dictionary['animals'][animal]['color']
                color = (color[2], color[1], color[0])
                cv2.drawContours(image=frame_temp,
                                 contours=self.animal_tracking_dictionary['animals'][animal]['contours'][frame_index],
                                 contourIdx=-1,
                                 color=color,
                                 thickness=1,
                                 offset=(
                                     int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0]),
                                     int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1]))
                                 )
                x_org = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0])
                y_org = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1]) - 5 if \
                    int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1]) - 5 > 0 else 0
                box_width = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][2] - \
                                self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0])
                box_height = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][3] - \
                                 self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1])
                score = round(
                    float(self.animal_tracking_dictionary["animals"][animal]["scores"][frame_index]) * 100, 2)
                text = f'Animal {self.animal_tracking_dictionary["animals"][animal]["animal_id"]}: ' \
                       f'{score}'
                animal_font_scale, animal_font_width = self.get_font_size(image_width=int(box_width),
                                                                          image_height=int(box_height))
                cv2.putText(img=frame_temp,
                            text=text,
                            org=(x_org, y_org),
                            color=color,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=animal_font_scale,
                            thickness=animal_font_width,
                            )
        return frame_temp

    def get_font_size(self, image_width, image_height, font_scale=2e-3, thickness_scale=5e-3):
        font_scale = (image_height + image_width) * font_scale
        font_scale = font_scale if font_scale > 0.5 else 0.5
        thickness = int(math.ceil(min(image_height, image_width) * thickness_scale))
        return font_scale, thickness


# Run the program
if __name__ == '__main__':
    app = wx.App()
    FrameGrabberInitialWindow('Frame Grabber')
    app.MainLoop()
