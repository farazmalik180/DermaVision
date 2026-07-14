# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package configurations
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code and build the React app
COPY . .
RUN npm run build

# Production serving stage using Nginx
FROM nginx:alpine

# Copy built files from the build stage to Nginx serve directory
COPY --from=build /app/dist /usr/share/nginx/html

# Expose the default Nginx port
EXPOSE 80

# Nginx starts automatically
CMD ["nginx", "-g", "daemon off;"]
