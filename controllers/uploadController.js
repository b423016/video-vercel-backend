export const uploadVideo = (req, res) => {
  try {
      if (!req.file) {
          return res.status(400).json({ error: 'No file uploaded.' });
      }
      const videoPath = `/uploads/${req.file.filename}`;
      res.status(200).json({ imageUrl: videoPath });
  } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({ error: 'Server error occurred during upload.' });
  }
};
