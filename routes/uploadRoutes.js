import { Router } from 'express';
const router = Router();
import { uploadFile } from '../controllers/uploadController';

router.post('/upload', uploadFile);

export default router;