function [px, py]=Compose(x, y, route, angle)
       angle=angle/180*pi;
       px=x+sum(route.*cos(angle));
       py=y+sum(route.*sin(angle));
end