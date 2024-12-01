import express from "express";
import registerUser from "../controller/registerController";
import loginUser from "../controller/loginController";

const routes = express.Router();

routes.post("/login", loginUser);
routes.post("/register", registerUser);

export default routes;
