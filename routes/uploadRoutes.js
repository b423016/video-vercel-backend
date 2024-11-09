import { Router } from 'express';
import multer from 'multer';
import path from 'path';
import { uploadVideo } from '../controllers/uploadController.js';

const router = Router();

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(path.resolve(), 'uploads'));
    },
    filename: (req, file, cb) => {
        cb(null, `${Date.now()}-${file.originalname}`);
    },
});

const upload = multer({ storage });

router.post('/', upload.single('video'), uploadVideo);

export default router;
