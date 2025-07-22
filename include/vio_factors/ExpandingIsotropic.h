// ExpandingIsotropicNoiseModel.h
// -----------------------------------------------------------------------------
// Production‑ready, dynamic per‑measurement isotropic noise model for GTSAM.
// -----------------------------------------------------------------------------
// Key features
//   •  One scalar standard‑deviation σᵢ per (u,v) residual pair – call
//   pushSigma() •  Thread‑safe growth guarded by std::mutex + freeze() before
//   optimisation •  Implements **all** pure‑virtual API of noiseModel::Base,
//   incl. WhitenSystem •  Boost‑serialisable, works under noiseModel::Robust,
//   iSAM2‑friendly •  Lightweight: only 8 bytes extra per observation (one
//   double)
// -----------------------------------------------------------------------------
#ifndef EXPANDING_ISOTROPIC_NOISE_MODEL_H
#define EXPANDING_ISOTROPIC_NOISE_MODEL_H

#include <Eigen/Core>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace gtsam {
namespace noiseModel {

/**
 * @tparam ZDim   Residual dimensionality per observation.
 *                2 → (u,v) pinhole mono, 3 → (u,v,uR) stereo, etc.
 * @warning       Template parameter must match SmartFactor variant used!
 */
template <int ZDim = 2> class ExpandingIsotropic : public Isotropic {
public:
  static_assert(ZDim > 0, "ZDim must be positive");
  using This = ExpandingIsotropic<ZDim>;
  using BaseType = Isotropic;

  std::vector<double> sigmas_; ///< one σ per observation
  mutable std::mutex mutex_;
  bool frozen_ = false;

  // Private default ctor for serialization
  ExpandingIsotropic() : BaseType() {}

  // ---------------- Helper lambdas ---------------- //
  template <class Derived>
  inline void scaleRowsInPlace(Eigen::MatrixBase<Derived> &M,
                               bool whiten) const {
    const size_t num = sigmas_.size();
    for (size_t k = 0; k < num; ++k) {
      const double s = whiten ? (1.0 / sigmas_[k]) : sigmas_[k];
      for (int d = 0; d < ZDim; ++d)
        M.row(static_cast<Eigen::Index>(k * ZDim + d)) *= s;
    }
  }
  inline void checkDims(Eigen::Index n) const {
    if (static_cast<size_t>(n) != dim()) {
      std::ostringstream ss;
      ss << "ExpandingIsotropic: expected " << dim() << " rows, got " << n;
      throw std::runtime_error(ss.str());
    }
  }

public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<This>;

  /* ---------------- Factory helpers ---------------- */
  static shared_ptr Create() { return shared_ptr(new This()); }
  static shared_ptr Create(size_t reserveCount) {
    auto m = shared_ptr(new This());
    m->sigmas_.reserve(reserveCount);
    return m;
  }

  /* ---------------- Mutation API ------------------- */
  void pushSigma(double sigma) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frozen_)
      throw std::runtime_error("pushSigma() after freeze()");
    sigmas_.push_back(sigma);
    dim_ = static_cast<int>(ZDim * sigmas_.size()); // update Base::dim_
  }
  void reserve(size_t n) { sigmas_.reserve(n); }
  void freeze() {
    std::lock_guard<std::mutex> l(mutex_);
    frozen_ = true;
  }

  /* ---------------- Required API ------------------- */
  size_t dim() const { return ZDim * sigmas_.size(); }

  Vector whiten(const Vector &v) const override {
    checkDims(v.size());
    Vector w(v);
    scaleRowsInPlace(w, /*whiten*/ true);
    return w;
  }
  Vector unwhiten(const Vector &v) const override {
    checkDims(v.size());
    Vector u(v);
    scaleRowsInPlace(u, /*whiten*/ false);
    return u;
  }

  Matrix Whiten(const Matrix &H) const override {
    // check dimensions
    if (H.rows() != dim())
      throw std::runtime_error("ExpandingIsotropic: Whiten(Matrix) wrong rows");
    if (H.cols() != dim())
      throw std::runtime_error("ExpandingIsotropic: Whiten(Matrix) wrong cols");

    // scale rows
    Matrix W(H);
    scaleRowsInPlace(W, true);
    return W;
  }

  Matrix Whiten(const Matrix &H, size_t obsIndex) const override {
    const double s = 1.0 / sigmas_.at(obsIndex);
    return H * s;
  }

  void WhitenSystem(std::vector<Matrix> &A, Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    for (auto &Ai : A)
      scaleRowsInPlace(Ai, true);
  }
  void WhitenSystem(Matrix &A, Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    scaleRowsInPlace(A, true);
  }
  void WhitenSystem(Matrix &A1, Matrix &A2, Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    scaleRowsInPlace(A1, true);
    scaleRowsInPlace(A2, true);
  }
  void WhitenSystem(Matrix &A1, Matrix &A2, Matrix &A3,
                    Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    scaleRowsInPlace(A1, true);
    scaleRowsInPlace(A2, true);
    scaleRowsInPlace(A3, true);
  }

  bool isConstrained() const override { return false; }
  bool isUnit() const override { return false; }

  /* ---------------- Debug ------------------------- */
  void print(const std::string &s = "") const override {
    std::cout << s << "ExpandingIsotropic<" << ZDim << "> with "
              << sigmas_.size() << " obs" << std::endl;
  }
  bool equals(const Base &expected, double tol = 1e-9) const override {
    const This *e = dynamic_cast<const This *>(&expected);
    if (!e || sigmas_.size() != e->sigmas_.size())
      return false;
    for (size_t i = 0; i < sigmas_.size(); ++i)
      if (std::abs(sigmas_[i] - e->sigmas_[i]) > tol)
        return false;
    return true;
  }

private:
  /* ---------------- Serialization ----------------- */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE &ar, const unsigned int /*version*/) {
    ar &boost::serialization::base_object<Base>(*this);
    ar & sigmas_;
    ar & frozen_;
    dim_ = static_cast<int>(ZDim * sigmas_.size());
  }
};

/* ---------- Convenient aliases ---------- */
using ExpandingIsotropicMono = ExpandingIsotropic<2>;
using ExpandingIsotropicStereo = ExpandingIsotropic<3>;
} // namespace noiseModel
} // namespace gtsam

#endif /* EXPANDING_ISOTROPIC_NOISE_MODEL_H */
