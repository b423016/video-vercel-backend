import fs from 'fs';
import path from 'path';

export const uploadVideo = (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: 'No video file uploaded.' });
        }

        const thumbnailCount = parseInt(req.body.thumbnailCount, 10);
        if (isNaN(thumbnailCount) || thumbnailCount < 1) {
            return res.status(400).json({ message: 'Invalid thumbnail count.' });
        }

        const uploadsDir = path.join(path.resolve(), 'uploads');
        const jsonFilePath = path.join(uploadsDir, `${Date.now()}-metadata.json`);

        const metadata = {
            num_thumbnails: thumbnailCount
        };

        fs.writeFileSync(jsonFilePath, JSON.stringify(metadata, null, 2));

        res.status(200).json({
            message: 'Video and metadata saved successfully!',
            videoPath: metadata.filePath,
            metadataPath: jsonFilePath,
        });
    } catch (error) {
        console.error('Error in uploadVideo:', error);
        res.status(500).json({ message: 'Server error while uploading video and metadata.' });
    }
};
