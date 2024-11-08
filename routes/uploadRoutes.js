import express from 'express';
import { handleVideoUpload, uploadMiddleware } from '../controllers/uploadController.js';

const router = express.Router();

// Route to handle video upload
router.post('/', uploadMiddleware, handleVideoUpload);

export default router;
