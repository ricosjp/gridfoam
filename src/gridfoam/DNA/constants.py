from gridfoam.DNA.enum import Axis

FACE_NEIGHBOR_MAP = {
     4: (Axis.Z, False), # -z face
    10: (Axis.Y, False), # -y face
    12: (Axis.X, False), # -x face
    14: (Axis.X, True),  # +x face
    16: (Axis.Y, True),  # +y face
    22: (Axis.Z, True),  # +z face
}

DOMAIN_BOUNDARY_MAP = {
    "domainZ-": 4,
    "domainY-": 10,
    "domainX-": 12,
    "domainX+": 14,
    "domainY+": 16,
    "domainZ+": 22,
}
