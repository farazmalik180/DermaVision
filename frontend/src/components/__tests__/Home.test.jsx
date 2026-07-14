import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Home from '../Home';

describe('Home Component', () => {
  it('renders the Home component correctly', () => {
    // Basic test to verify that the home component mounts without crashing
    render(<Home />);
    
    // We expect some text or heading to be present
    // You may need to update this text based on the actual content in Home.jsx
    const uploadText = screen.getByText(/upload/i);
    expect(uploadText).toBeInTheDocument();
  });
});
