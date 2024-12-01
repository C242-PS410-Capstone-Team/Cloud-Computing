// Function to validate email
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Function to validate password
export const validatePassword = (password: string): true | string => {
  const minLength = 8; // Minimum length for the password
  const passwordCriteria = [
    {
      regex: /[A-Z]/,
      message: "Password must contain at least one uppercase letter.",
    },
    {
      regex: /[a-z]/,
      message: "Password must contain at least one lowercase letter.",
    },
    { regex: /\d/, message: "Password must contain at least one number." },
    {
      regex: /[!@#$%^&*(),.?":{}|<>]/,
      message: "Password must contain at least one special character.",
    },
  ];

  if (password.length < minLength) {
    return `Password must be at least ${minLength} characters.`;
  }

  for (const { regex, message } of passwordCriteria) {
    if (!regex.test(password)) {
      return message;
    }
  }

  return true;
};
