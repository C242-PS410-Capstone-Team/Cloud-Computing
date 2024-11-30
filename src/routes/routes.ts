import express from "express";
import registerUser from "../controller/registerController";

const routes = express.Router();

routes.post("/register", registerUser);

export default routes;
