// ExpandingIsotropicNoiseModel.h  (with DCS + scores)
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
#include <algorithm>
#include <cmath>

namespace gtsam {
namespace noiseModel {

template <int ZDim = 2>
class ExpandingIsotropic : public Isotropic {
public:
  static_assert(ZDim > 0, "ZDim must be positive");
  using This = ExpandingIsotropic<ZDim>;
  using BaseType = Isotropic;

  /* -------- Stored per-observation state -------- */
  std::vector<double> sigmas_;   ///< std-dev σ_i per observation (pixels)
  std::vector<double> scores_;   ///< raw scores s_i in [0,1] (optional)
  std::vector<double> phis_;     ///< DCS φ_i per observation (derived or pushed)

  mutable std::mutex mutex_;
  bool frozen_ = false;

  /* -------- DCS controls -------- */
  bool use_dcs_ = true;          ///< enable/disable DCS inside WhitenSystem
  double dcs_phi_min_ = 0.5;     ///< mapping floor for φ(s)
  double dcs_phi_max_ = 5.0;     ///< mapping ceil  for φ(s)
  double dcs_gamma_   = 1.5;     ///< mapping exponent

  ExpandingIsotropic() : BaseType() {}

  /* ---------------- Helpers ---------------- */
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

  template <class Derived>
  inline void scaleRowsByObsGains(Eigen::MatrixBase<Derived> &M,
                                  const std::vector<double> &gains) const {
    const size_t num = sigmas_.size();
    if (gains.size() != num)
      throw std::runtime_error("ExpandingIsotropic: DCS gains size mismatch");
    for (size_t k = 0; k < num; ++k) {
      const double g = gains[k];
      for (int d = 0; d < ZDim; ++d)
        M.row(static_cast<Eigen::Index>(k * ZDim + d)) *= g;
    }
  }

  inline void checkDims(Eigen::Index n) const {
    if (static_cast<size_t>(n) != dim()) {
      std::ostringstream ss;
      ss << "ExpandingIsotropic: expected " << dim() << " rows, got " << n;
      throw std::runtime_error(ss.str());
    }
  }

  /* Compute sqrt(s_k) DCS gains from a whitened residual vector (per obs). */
  inline std::vector<double> dcsSqrtGainsFromWhitenedResidual(const Vector &b_whitened) const {
    const size_t num = sigmas_.size();
    std::vector<double> gains(num, 1.0);
    if (!use_dcs_) return gains;

    for (size_t k = 0; k < num; ++k) {
      // e_k = sum over the ZDim rows of whitened residual^2 for obs k
      double ek = 0.0;
      const size_t row0 = k * ZDim;
      for (int d = 0; d < ZDim; ++d) {
        const double r = b_whitened(static_cast<Eigen::Index>(row0 + d));
        ek += r * r;
      }
      // choose φ for this obs (prefer explicit phis_, else map from score, else default)
      double phi = 1.0;
      if (k < phis_.size()) {
        phi = phis_[k];
      } else if (k < scores_.size()) {
        const double s = std::clamp(scores_[k], 0.0, 1.0);
        phi = dcs_phi_min_ + (dcs_phi_max_ - dcs_phi_min_) * std::pow(s, dcs_gamma_);
      } else {
        phi = 1.0; // sane default
      }
      // DCS scaling s(e) = min(1, 2φ / (φ + e))
      const double scale = std::min(1.0, (2.0 * phi) / (phi + ek + 1e-12));
      gains[k] = std::sqrt(std::max(0.0, scale));
    }
    return gains;
  }

public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<This>;

  /* ---------------- Factory helpers ---------------- */
  static shared_ptr Create() { return shared_ptr(new This()); }
  static shared_ptr Create(size_t reserveCount) {
    auto m = shared_ptr(new This());
    m->sigmas_.reserve(reserveCount);
    m->scores_.reserve(reserveCount);
    m->phis_.reserve(reserveCount);
    return m;
  }

  /* ---------------- Mutation API ------------------- */
  void pushSigma(double sigma) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frozen_) throw std::runtime_error("pushSigma() after freeze()");
    if (!(sigma > 0.0 && std::isfinite(sigma)))
      throw std::runtime_error("pushSigma(): sigma must be finite and > 0");
    sigmas_.push_back(sigma);
    dim_ = static_cast<int>(ZDim * sigmas_.size()); // update Base::dim_
  }

  void pushScore(double score01) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frozen_) throw std::runtime_error("pushScore() after freeze()");
    const double s = std::clamp(score01, 0.0, 1.0);
    scores_.push_back(s);
  }

  void pushPhi(double phi) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frozen_) throw std::runtime_error("pushPhi() after freeze()");
    const double p = std::max(1e-9, phi);
    phis_.push_back(p);
  }

  void reserve(size_t n) {
    sigmas_.reserve(n);
    scores_.reserve(n);
    phis_.reserve(n);
  }

  void freeze() {
    std::lock_guard<std::mutex> l(mutex_);
    frozen_ = true;
  }

  /* DCS configuration */
  void enableDCS(bool enabled) { use_dcs_ = enabled; }
  void setDCSMapping(double phi_min, double phi_max, double gamma = 1.5) {
    if (!(phi_min > 0 && phi_max > phi_min)) throw std::runtime_error("setDCSMapping: bad phi range");
    if (!(gamma > 0)) throw std::runtime_error("setDCSMapping: gamma must be > 0");
    dcs_phi_min_ = phi_min;
    dcs_phi_max_ = phi_max;
    dcs_gamma_   = gamma;
  }

  /* ---------------- Required API ------------------- */
  size_t dim() const { return ZDim * sigmas_.size(); }

  Vector whiten(const Vector &v) const override {
    checkDims(v.size());
    Vector w(v);
    scaleRowsInPlace(w, /*whiten*/ true);
    return w; // DCS not applied here (no residual context)
  }

  Vector unwhiten(const Vector &v) const override {
    checkDims(v.size());
    Vector u(v);
    scaleRowsInPlace(u, /*whiten*/ false);
    return u;
  }

  Matrix Whiten(const Matrix &H) const override {
    if (H.rows() != dim())
      throw std::runtime_error("ExpandingIsotropic: Whiten(Matrix) wrong rows");
    if (H.cols() != dim())
      throw std::runtime_error("ExpandingIsotropic: Whiten(Matrix) wrong cols");
    Matrix W(H);
    scaleRowsInPlace(W, true);
    return W; // DCS not applied (no residual vector)
  }

  Matrix Whiten(const Matrix &H, size_t obsIndex) const override {
    const double s = 1.0 / sigmas_.at(obsIndex);
    return H * s;
  }

  /* --------- DCS-aware WhitenSystem overloads --------- */
  void WhitenSystem(std::vector<Matrix> &A, Vector &b) const override {
    checkDims(b.size());
    // 1) standard σ-whitening
    scaleRowsInPlace(b, true);
    for (auto &Ai : A) scaleRowsInPlace(Ai, true);

    // 2) DCS scaling (requires residuals)
    if (use_dcs_) {
      const auto gains = dcsSqrtGainsFromWhitenedResidual(b);
      // multiply each obs block (rows) by sqrt(s_k)
      for (auto &Ai : A) scaleRowsByObsGains(Ai, gains);
      scaleRowsByObsGains(b, gains);
    }
  }

  void WhitenSystem(Matrix &A, Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    scaleRowsInPlace(A, true);

    if (use_dcs_) {
      const auto gains = dcsSqrtGainsFromWhitenedResidual(b);
      scaleRowsByObsGains(A, gains);
      scaleRowsByObsGains(b, gains);
    }
  }

  void WhitenSystem(Matrix &A1, Matrix &A2, Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    scaleRowsInPlace(A1, true);
    scaleRowsInPlace(A2, true);

    if (use_dcs_) {
      const auto gains = dcsSqrtGainsFromWhitenedResidual(b);
      scaleRowsByObsGains(A1, gains);
      scaleRowsByObsGains(A2, gains);
      scaleRowsByObsGains(b, gains);
    }
  }

  void WhitenSystem(Matrix &A1, Matrix &A2, Matrix &A3, Vector &b) const override {
    checkDims(b.size());
    scaleRowsInPlace(b, true);
    scaleRowsInPlace(A1, true);
    scaleRowsInPlace(A2, true);
    scaleRowsInPlace(A3, true);

    if (use_dcs_) {
      const auto gains = dcsSqrtGainsFromWhitenedResidual(b);
      scaleRowsByObsGains(A1, gains);
      scaleRowsByObsGains(A2, gains);
      scaleRowsByObsGains(A3, gains);
      scaleRowsByObsGains(b, gains);
    }
  }

  bool isConstrained() const override { return false; }
  bool isUnit() const override { return false; }

  /* ---------------- Debug ------------------------- */
  void print(const std::string &s = "") const override {
    std::cout << s << "ExpandingIsotropic<" << ZDim << "> with "
              << sigmas_.size() << " obs"
              << "  DCS:" << (use_dcs_ ? "on" : "off")
              << std::endl;
  }

  bool equals(const Base &expected, double tol = 1e-9) const override {
    const This *e = dynamic_cast<const This *>(&expected);
    if (!e || sigmas_.size() != e->sigmas_.size()) return false;
    for (size_t i = 0; i < sigmas_.size(); ++i)
      if (std::abs(sigmas_[i] - e->sigmas_[i]) > tol) return false;
    // scores/phis are auxiliary; ignore in equality
    return true;
  }

private:
  /* ---------------- Serialization ----------------- */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE &ar, const unsigned int /*version*/) {
    ar &boost::serialization::base_object<Base>(*this);
    ar & sigmas_;
    ar & scores_;
    ar & phis_;
    ar & frozen_;
    ar & use_dcs_;
    ar & dcs_phi_min_;
    ar & dcs_phi_max_;
    ar & dcs_gamma_;
    dim_ = static_cast<int>(ZDim * sigmas_.size());
  }
};

/* ---------- Convenient aliases ---------- */
using ExpandingIsotropicMono   = ExpandingIsotropic<2>;
using ExpandingIsotropicStereo = ExpandingIsotropic<3>;

} // namespace noiseModel
} // namespace gtsam

#endif /* EXPANDING_ISOTROPIC_NOISE_MODEL_H */
